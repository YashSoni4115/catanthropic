
package game;

import board.Board;
import board.EdgeLocation;
import board.Location;
import board.Road;
import board.Structure;
import board.Tile;
import board.VertexLocation;
import java.awt.Color;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.*;

/**
 * SelfPlayGameRunner generates a single legal Catan self-play game trajectory.
 * Each turn applies random legal actions supported by this codebase.
 * Outputs JSONL, one line per state snapshot, for ML dataset generation.
 */
public class SelfPlayGameRunner {
    private static final int DEFAULT_MAX_TURNS = 1200;
    private static final String[] RES_TYPES = {"BRICK", "WOOL", "ORE", "GRAIN", "LUMBER"};

    private static final class Config {
        int maxTurns = DEFAULT_MAX_TURNS;
        boolean debug = false;
        boolean strict = true;
        long seed = 0L;
        boolean hasSeed = false;
        int maxConsecutivePassTurns = 360;
        int maxNoProgressTurns = 520;
        boolean guided = false;
        String guidedModelPath = "";
        String guidedPythonExec = "python3";
    }

    private static final class GuidedModelScorer {
        private final Process process;
        private final BufferedWriter input;
        private final BufferedReader output;

        GuidedModelScorer(Config config) throws IOException {
            ArrayList<String> command = new ArrayList<String>();
            command.add(config.guidedPythonExec);
            command.add("-u");
            command.add("ai/selfplay/infer_win_model_v1.py");
            command.add("--model");
            command.add(config.guidedModelPath);

            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectError(ProcessBuilder.Redirect.INHERIT);
            this.process = pb.start();
            this.input = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
            this.output = new BufferedReader(new InputStreamReader(process.getInputStream()));
        }

        List<Double> score(List<double[]> vectors) throws IOException {
            String payload = vectorsJson(vectors);
            input.write(payload);
            input.newLine();
            input.flush();

            String response = output.readLine();
            if (response == null) {
                throw new IOException("guided scorer returned EOF");
            }
            return parseProbabilities(response, vectors.size());
        }

        void close() {
            try {
                input.close();
            } catch (Exception ignored) {
            }
            try {
                output.close();
            } catch (Exception ignored) {
            }
            try {
                process.destroy();
            } catch (Exception ignored) {
            }
        }
    }

    private static final class ActionTrace {
        ArrayList<String> successfulActions = new ArrayList<String>();
    }

    public static void main(String[] args) {
        Config config = parseArgs(args);
        String gameId = UUID.randomUUID().toString();
        Random rng = config.hasSeed ? new Random(config.seed) : new Random();
        ArrayList<Player> players = makePlayers();
        Game game = new Game(players);
        List<String> trajectory = new ArrayList<>();
        int turnIndex = 0;
        int consecutivePassTurns = 0;
        int noProgressTurns = 0;
        HashSet<Integer> seenTurns = new HashSet<Integer>();
        String terminationReason = "error";
        GuidedModelScorer scorer = null;

        try {
            if (config.guided) {
                if (config.guidedModelPath == null || config.guidedModelPath.trim().isEmpty()) {
                    throw new IllegalArgumentException("--guided requires --guided-model-path");
                }
                scorer = new GuidedModelScorer(config);
            }

            runInitialPlacements(game, players, rng);
            assertAllResourcesNonNegative(players, "post_initial_placement");

            while (!game.over() && turnIndex < config.maxTurns) {
                if (!seenTurns.add(Integer.valueOf(turnIndex))) {
                    fail(config, "duplicate_turn_index", "turn_index=" + turnIndex);
                }

                Player current = players.get(turnIndex % players.size());
                int currentPlayerId = turnIndex % players.size();
                int totalVpBefore = totalVictoryPoints(players);
                int[] currentResBeforeTurn = resourceVector(current);
                int totalResBeforeTurn = totalResources(players);
                String fingerprintBefore = stateFingerprint(players);

                if (config.debug) {
                    System.err.println("[TURN_START] game_id=" + gameId
                        + " turn=" + turnIndex
                        + " player_id=" + currentPlayerId
                        + " player_name=" + current.getName()
                        + " vp_before=" + totalVpBefore
                        + " total_res_before=" + totalResBeforeTurn
                        + " state_fp_before=" + fingerprintBefore);
                }

                int rollApplications = 0;
                int roll = game.roll(current);
                rollApplications++;
                if (roll == 7) {
                    safeHalfCards(players, rng);
                    maybeMoveRobber(game, rng);
                }
                if (rollApplications != 1) {
                    fail(config, "roll_applied_invalid_count", "count=" + rollApplications + " turn=" + turnIndex);
                }

                int totalResAfterRoll = totalResources(players);

                ActionTrace trace = performRandomTurnActions(game, players, current, rng, config, gameId, turnIndex, roll, scorer);
                ArrayList<String> actions = trace.successfulActions;

                String actionLabel = actions.isEmpty() ? "pass" : String.join("+", actions);
                trajectory.add(snapshotJsonLine(game, players, gameId, turnIndex, currentPlayerId, current.getName(), actionLabel, actions, roll));

                assertAllResourcesNonNegative(players, "post_turn:" + turnIndex);

                int totalVpAfter = totalVictoryPoints(players);
                int totalResAfterTurn = totalResources(players);
                String fingerprintAfter = stateFingerprint(players);
                boolean passOnly = isPassOnly(actions);

                if (passOnly) {
                    consecutivePassTurns++;
                } else {
                    consecutivePassTurns = 0;
                }
                if (totalVpAfter == totalVpBefore) {
                    noProgressTurns++;
                } else {
                    noProgressTurns = 0;
                }

                if (config.debug) {
                    System.err.println("[TURN_END] game_id=" + gameId
                        + " turn=" + turnIndex
                        + " roll=" + roll
                        + " total_res_after_roll=" + totalResAfterRoll
                        + " total_res_after_turn=" + totalResAfterTurn
                        + " vp_before=" + totalVpBefore
                        + " vp_after=" + totalVpAfter
                        + " current_res_before=" + resourceVectorString(currentResBeforeTurn)
                        + " current_res_after=" + resourceVectorString(resourceVector(current))
                        + " pass_only=" + passOnly
                        + " no_progress_turns=" + noProgressTurns
                        + " consecutive_pass_turns=" + consecutivePassTurns
                        + " state_fp_after=" + fingerprintAfter);
                }

                if (config.strict && consecutivePassTurns > config.maxConsecutivePassTurns) {
                    fail(config, "degenerate_pass_loop", "consecutive_pass_turns=" + consecutivePassTurns + " turn=" + turnIndex);
                }
                if (config.strict && noProgressTurns > config.maxNoProgressTurns) {
                    fail(config, "no_progress_too_long", "no_progress_turns=" + noProgressTurns + " turn=" + turnIndex);
                }

                turnIndex++;
            }

            Player winner = game.winningPlayer();
            if (game.over()) {
                if (winner == null) {
                    fail(config, "terminal_without_winner", "game.over() true but winningPlayer null");
                }
                terminationReason = "winner";
            } else if (turnIndex >= config.maxTurns) {
                terminationReason = "truncated_max_turns";
            } else {
                terminationReason = "error";
            }

            String winnerId = "";
            String winnerName = "";
            if ("winner".equals(terminationReason) && winner != null) {
                winnerId = winner.getName();
                winnerName = winner.getName();
            }

            for (int i = 0; i < trajectory.size(); i++) {
                boolean isTerminal = i == trajectory.size() - 1;
                String suffix = ",\"winner_player_id\":\"" + escape(winnerId) + "\",\"winner_player_name\":\""
                    + escape(winnerName) + "\",\"termination_reason\":\"" + escape(terminationReason)
                    + "\",\"simulation_error\":\"\",\"is_terminal\":" + isTerminal + "}";
                String base = trajectory.get(i);
                if (base.endsWith("}")) {
                    base = base.substring(0, base.length() - 1);
                }
                System.out.println(base + suffix);
            }
        } catch (Exception e) {
            terminationReason = "error";
            System.err.println("Error during self-play simulation: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        } finally {
            if (scorer != null) {
                scorer.close();
            }
        }
    }

    private static Config parseArgs(String[] args) {
        Config config = new Config();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            if ("--debug".equals(arg)) {
                config.debug = true;
            } else if ("--no-strict".equals(arg)) {
                config.strict = false;
            } else if ("--max-turns".equals(arg) && i + 1 < args.length) {
                config.maxTurns = Integer.parseInt(args[++i]);
            } else if ("--seed".equals(arg) && i + 1 < args.length) {
                config.seed = Long.parseLong(args[++i]);
                config.hasSeed = true;
            } else if ("--max-consecutive-pass-turns".equals(arg) && i + 1 < args.length) {
                config.maxConsecutivePassTurns = Integer.parseInt(args[++i]);
            } else if ("--max-no-progress-turns".equals(arg) && i + 1 < args.length) {
                config.maxNoProgressTurns = Integer.parseInt(args[++i]);
            } else if ("--guided".equals(arg)) {
                config.guided = true;
            } else if ("--guided-model-path".equals(arg) && i + 1 < args.length) {
                config.guidedModelPath = args[++i];
            } else if ("--guided-python-exec".equals(arg) && i + 1 < args.length) {
                config.guidedPythonExec = args[++i];
            }
        }
        return config;
    }

    private static ArrayList<Player> makePlayers() {
        ArrayList<Player> players = new ArrayList<>();
        players.add(new Player("P0", Color.ORANGE));
        players.add(new Player("P1", Color.RED));
        players.add(new Player("P2", Color.BLUE));
        players.add(new Player("P3", Color.WHITE));
        return players;
    }

    private static void runInitialPlacements(Game game, ArrayList<Player> players, Random rng) {
        for (Player player : players) {
            placeRandomInitialSettlementAndRoad(game, player, rng);
        }
        for (int i = players.size() - 1; i >= 0; i--) {
            placeRandomInitialSettlementAndRoad(game, players.get(i), rng);
        }
    }

    private static void placeRandomInitialSettlementAndRoad(Game game, Player player, Random rng) {
        List<VertexLocation> vertices = allVertexLocations();
        Collections.shuffle(vertices, rng);
        for (VertexLocation vertex : vertices) {
            if (game.getBoard().placeStructureNoRoad(vertex, player)) {
                placeRandomRoad(game, player, rng);
                return;
            }
        }
    }

    private static ActionTrace performRandomTurnActions(
        Game game,
        ArrayList<Player> players,
        Player player,
        Random rng,
        Config config,
        String gameId,
        int turnIndex,
        int roll,
        GuidedModelScorer scorer
    ) {
        ActionTrace trace = new ActionTrace();
        int actionBudget = 2 + rng.nextInt(3);
        for (int i = 0; i < actionBudget; i++) {
            List<String> candidates = legalActionKinds(game, player);
            if (candidates.isEmpty()) {
                break;
            }
            String chosen = chooseActionKind(game, players, player, candidates, rng, config, turnIndex, roll, scorer);
            boolean success = applyActionKind(game, player, chosen, rng, config, gameId, turnIndex, i);
            if (success) {
                trace.successfulActions.add(chosen);
                if ("pass".equals(chosen)) {
                    break;
                }
            }
        }
        return trace;
    }

    private static List<String> legalActionKinds(Game game, Player player) {
        ArrayList<String> actions = new ArrayList<>();
        if (canBuildRoad(player) && hasAnyLegalRoadPlacement(game, player)) {
            actions.add("build_road");
        }
        if (canBuildSettlement(player) && hasAnyLegalSettlementPlacement(game, player)) {
            actions.add("build_settlement");
        }
        if (canBuildCity(player) && hasAnyLegalCityPlacement(game, player)) {
            actions.add("build_city");
        }
        if (canBuyDevCard(player) && !game.getDeck().isEmpty()) {
            actions.add("buy_dev_card");
        }
        if (canDoBankTrade(player)) {
            actions.add("trade_bank");
        }
        actions.add("pass");
        return actions;
    }

    private static String chooseActionKind(
        Game game,
        ArrayList<Player> players,
        Player player,
        List<String> candidates,
        Random rng,
        Config config,
        int turnIndex,
        int roll,
        GuidedModelScorer scorer
    ) {
        String guidedChoice = chooseGuidedActionKind(players, player, candidates, config, turnIndex, roll, scorer);
        if (guidedChoice != null) {
            return guidedChoice;
        }

        if (candidates.contains("build_city")) {
            return "build_city";
        }
        if (candidates.contains("build_settlement")) {
            return "build_settlement";
        }
        if (candidates.contains("build_road") && bestRoadExpansionScore(game, player) >= 2) {
            return "build_road";
        }
        if (candidates.contains("trade_bank")) {
            return "trade_bank";
        }

        if (candidates.size() > 1 && candidates.contains("pass")) {
            boolean hasNonPass = false;
            for (String action : candidates) {
                if (!"pass".equals(action)) {
                    hasNonPass = true;
                    break;
                }
            }
            if (hasNonPass && rng.nextDouble() < 0.85) {
                ArrayList<String> filtered = new ArrayList<String>();
                for (String action : candidates) {
                    if (!"pass".equals(action)) {
                        filtered.add(action);
                    }
                }
                candidates = filtered;
            }
        }

        double total = 0.0;
        ArrayList<Double> weights = new ArrayList<Double>();
        for (String action : candidates) {
            double weight = actionWeight(game, player, action);
            weights.add(Double.valueOf(weight));
            total += weight;
        }

        if (total <= 0.0) {
            return candidates.get(rng.nextInt(candidates.size()));
        }

        double pick = rng.nextDouble() * total;
        double cumulative = 0.0;
        for (int i = 0; i < candidates.size(); i++) {
            cumulative += weights.get(i).doubleValue();
            if (pick <= cumulative) {
                return candidates.get(i);
            }
        }
        return candidates.get(candidates.size() - 1);
    }

    private static String chooseGuidedActionKind(
        ArrayList<Player> players,
        Player player,
        List<String> candidates,
        Config config,
        int turnIndex,
        int roll,
        GuidedModelScorer scorer
    ) {
        if (config == null || !config.guided || scorer == null || players == null || turnIndex < 0) {
            return null;
        }

        try {
            double[] base = baseFeatureVector(players, player, turnIndex, roll);
            ArrayList<double[]> projected = new ArrayList<double[]>();
            for (String action : candidates) {
                double[] v = Arrays.copyOf(base, base.length);
                applyProjectedActionDelta(v, action);
                projected.add(v);
            }
            List<Double> scores = scorer.score(projected);
            if (scores.size() != candidates.size()) {
                return null;
            }

            int best = 0;
            double bestScore = scores.get(0).doubleValue();
            for (int i = 1; i < scores.size(); i++) {
                double score = scores.get(i).doubleValue();
                if (score > bestScore) {
                    best = i;
                    bestScore = score;
                }
            }
            return candidates.get(best);
        } catch (Exception ignored) {
            return null;
        }
    }

    private static double[] baseFeatureVector(ArrayList<Player> players, Player current, int turnIndex, int roll) {
        double[] v = new double[24];
        v[0] = turnIndex;
        v[1] = 0.0;
        v[2] = 0.0;
        v[3] = 1.0;
        v[4] = 0.0;
        v[5] = 0.0;
        v[6] = 0.0;
        v[7] = roll;

        double currentVp = current.getVictoryPoints();
        double currentBrick = current.getNumberResourcesType("BRICK");
        double currentWool = current.getNumberResourcesType("WOOL");
        double currentOre = current.getNumberResourcesType("ORE");
        double currentGrain = current.getNumberResourcesType("GRAIN");
        double currentLumber = current.getNumberResourcesType("LUMBER");

        v[8] = currentVp;
        v[10] = currentBrick;
        v[11] = currentWool;
        v[12] = currentOre;
        v[13] = currentGrain;
        v[14] = currentLumber;
        v[9] = currentBrick + currentWool + currentOre + currentGrain + currentLumber;

        ArrayList<Player> opponents = new ArrayList<Player>();
        for (Player p : players) {
            if (!p.getName().equals(current.getName())) {
                opponents.add(p);
            }
        }
        Collections.sort(opponents, new Comparator<Player>() {
            public int compare(Player a, Player b) {
                return a.getName().compareTo(b.getName());
            }
        });

        for (int i = 0; i < 3; i++) {
            int vpIdx = 15 + (i * 2);
            int resIdx = vpIdx + 1;
            if (i < opponents.size()) {
                Player op = opponents.get(i);
                v[vpIdx] = op.getVictoryPoints();
                v[resIdx] = op.getNumberResourcesType("BRICK")
                    + op.getNumberResourcesType("WOOL")
                    + op.getNumberResourcesType("ORE")
                    + op.getNumberResourcesType("GRAIN")
                    + op.getNumberResourcesType("LUMBER");
            } else {
                v[vpIdx] = 0.0;
                v[resIdx] = 0.0;
            }
        }

        refreshDiffFeatures(v);
        return v;
    }

    private static void applyProjectedActionDelta(double[] v, String action) {
        if ("build_city".equals(action)) {
            v[8] += 1.0;
            v[12] -= 3.0;
            v[13] -= 2.0;
        } else if ("build_settlement".equals(action)) {
            v[8] += 1.0;
            v[10] -= 1.0;
            v[11] -= 1.0;
            v[13] -= 1.0;
            v[14] -= 1.0;
        } else if ("build_road".equals(action)) {
            v[10] -= 1.0;
            v[14] -= 1.0;
        } else if ("buy_dev_card".equals(action)) {
            v[11] -= 1.0;
            v[12] -= 1.0;
            v[13] -= 1.0;
        } else if ("trade_bank".equals(action)) {
            int hi = 10;
            int lo = 10;
            for (int i = 11; i <= 14; i++) {
                if (v[i] > v[hi]) {
                    hi = i;
                }
                if (v[i] < v[lo]) {
                    lo = i;
                }
            }
            v[hi] -= 4.0;
            v[lo] += 1.0;
        }

        for (int i = 10; i <= 14; i++) {
            if (v[i] < 0.0) {
                v[i] = 0.0;
            }
        }
        v[9] = v[10] + v[11] + v[12] + v[13] + v[14];
        refreshDiffFeatures(v);
    }

    private static void refreshDiffFeatures(double[] v) {
        double oppVpMax = Math.max(v[15], Math.max(v[17], v[19]));
        double oppVpMean = (v[15] + v[17] + v[19]) / 3.0;
        double oppResMean = (v[16] + v[18] + v[20]) / 3.0;
        v[21] = v[8] - oppVpMax;
        v[22] = v[8] - oppVpMean;
        v[23] = v[9] - oppResMean;
    }

    private static String vectorsJson(List<double[]> vectors) {
        StringBuilder sb = new StringBuilder();
        sb.append("{\"vectors\":[");
        for (int i = 0; i < vectors.size(); i++) {
            double[] v = vectors.get(i);
            sb.append("[");
            for (int j = 0; j < v.length; j++) {
                sb.append(v[j]);
                if (j < v.length - 1) {
                    sb.append(",");
                }
            }
            sb.append("]");
            if (i < vectors.size() - 1) {
                sb.append(",");
            }
        }
        sb.append("]}");
        return sb.toString();
    }

    private static List<Double> parseProbabilities(String response, int expectedCount) {
        int left = response.indexOf("[");
        int right = response.indexOf("]", left + 1);
        if (left < 0 || right < 0) {
            throw new IllegalStateException("invalid scorer response: " + response);
        }
        String body = response.substring(left + 1, right).trim();
        ArrayList<Double> out = new ArrayList<Double>();
        if (body.isEmpty()) {
            return out;
        }
        String[] parts = body.split(",");
        for (String part : parts) {
            out.add(Double.valueOf(Double.parseDouble(part.trim())));
        }
        if (out.size() != expectedCount) {
            throw new IllegalStateException("scorer size mismatch expected=" + expectedCount + " got=" + out.size());
        }
        return out;
    }

    private static double actionWeight(Game game, Player player, String action) {
        if ("build_settlement".equals(action)) {
            return 8.5;
        }
        if ("build_city".equals(action)) {
            return 8.0;
        }
        if ("build_road".equals(action)) {
            int expansionScore = bestRoadExpansionScore(game, player);
            return 2.5 + (expansionScore * 1.2);
        }
        if ("buy_dev_card".equals(action)) {
            return 4.5;
        }
        if ("trade_bank".equals(action)) {
            return 3.8;
        }
        if ("pass".equals(action)) {
            return 0.25;
        }
        return 1.0;
    }

    private static boolean applyActionKind(
        Game game,
        Player player,
        String action,
        Random rng,
        Config config,
        String gameId,
        int turnIndex,
        int actionIndex
    ) {
        int[] before = resourceVector(player);

        if ("pass".equals(action)) {
            int[] afterPass = resourceVector(player);
            assertResourceVectorEquals(before, afterPass, "pass mutated resources unexpectedly");
            logAttempt(config, gameId, turnIndex, actionIndex, player, action, true, "", before, afterPass);
            return true;
        }

        if ("build_road".equals(action)) {
            if (!canBuildRoad(player)) {
                int[] after = resourceVector(player);
                logAttempt(config, gameId, turnIndex, actionIndex, player, action, false, "precheck_canBuildRoad_false", before, after);
                return false;
            }
            if (placeRandomRoad(game, player, rng)) {
                int buyCode = game.buyRoad(player);
                int[] after = resourceVector(player);
                if (buyCode != 0) {
                    fail(config, "buyRoad_failed_after_place", "buyCode=" + buyCode + " turn=" + turnIndex);
                }
                assertResourceDelta(after, before, new int[] {-1, 0, 0, 0, -1}, "build_road");
                logAttempt(config, gameId, turnIndex, actionIndex, player, action, true, "", before, after);
                return true;
            }
            int[] after = resourceVector(player);
            assertResourceVectorEquals(before, after, "build_road failed but mutated resources");
            logAttempt(config, gameId, turnIndex, actionIndex, player, action, false, "no_legal_road_location", before, after);
            return false;
        }

        if ("build_settlement".equals(action)) {
            if (!canBuildSettlement(player)) {
                int[] after = resourceVector(player);
                logAttempt(config, gameId, turnIndex, actionIndex, player, action, false, "precheck_canBuildSettlement_false", before, after);
                return false;
            }
            if (placeRandomSettlement(game, player, rng)) {
                int buyCode = game.buySettlement(player);
                int[] after = resourceVector(player);
                if (buyCode != 0) {
                    fail(config, "buySettlement_failed_after_place", "buyCode=" + buyCode + " turn=" + turnIndex);
                }
                assertResourceDelta(after, before, new int[] {-1, -1, 0, -1, -1}, "build_settlement");
                logAttempt(config, gameId, turnIndex, actionIndex, player, action, true, "", before, after);
                return true;
            }
            int[] after = resourceVector(player);
            assertResourceVectorEquals(before, after, "build_settlement failed but mutated resources");
            logAttempt(config, gameId, turnIndex, actionIndex, player, action, false, "no_legal_settlement_location", before, after);
            return false;
        }

        if ("build_city".equals(action)) {
            if (!canBuildCity(player)) {
                int[] after = resourceVector(player);
                logAttempt(config, gameId, turnIndex, actionIndex, player, action, false, "precheck_canBuildCity_false", before, after);
                return false;
            }
            if (placeRandomCity(game, player, rng)) {
                int buyCode = game.buyCity(player);
                int[] after = resourceVector(player);
                if (buyCode != 0) {
                    fail(config, "buyCity_failed_after_place", "buyCode=" + buyCode + " turn=" + turnIndex);
                }
                assertResourceDelta(after, before, new int[] {0, 0, -3, -2, 0}, "build_city");
                logAttempt(config, gameId, turnIndex, actionIndex, player, action, true, "", before, after);
                return true;
            }
            int[] after = resourceVector(player);
            assertResourceVectorEquals(before, after, "build_city failed but mutated resources");
            logAttempt(config, gameId, turnIndex, actionIndex, player, action, false, "no_upgradeable_settlement", before, after);
            return false;
        }

        if ("buy_dev_card".equals(action)) {
            int buyCode = game.buyDevCard(player);
            int[] after = resourceVector(player);
            if (buyCode == 0) {
                assertResourceDelta(after, before, new int[] {0, -1, -1, -1, 0}, "buy_dev_card");
                logAttempt(config, gameId, turnIndex, actionIndex, player, action, true, "", before, after);
                return true;
            }
            assertResourceVectorEquals(before, after, "buy_dev_card failed but mutated resources");
            logAttempt(config, gameId, turnIndex, actionIndex, player, action, false, "buy_dev_card_code_" + buyCode, before, after);
            return false;
        }

        if ("trade_bank".equals(action)) {
            boolean success = performBankTrade(game, player);
            int[] after = resourceVector(player);
            if (success) {
                logAttempt(config, gameId, turnIndex, actionIndex, player, action, true, "", before, after);
                return true;
            }
            assertResourceVectorEquals(before, after, "trade_bank failed but mutated resources");
            logAttempt(config, gameId, turnIndex, actionIndex, player, action, false, "no_valid_bank_trade", before, after);
            return false;
        }

        int[] afterUnknown = resourceVector(player);
        logAttempt(config, gameId, turnIndex, actionIndex, player, action, false, "unknown_action", before, afterUnknown);
        return false;
    }

    private static boolean canBuildRoad(Player player) {
        return player.getNumberResourcesType("BRICK") >= 1
            && player.getNumberResourcesType("LUMBER") >= 1
            && player.getNumbRoads() < 15;
    }

    private static boolean canBuildSettlement(Player player) {
        return player.getNumberResourcesType("BRICK") >= 1
            && player.getNumberResourcesType("LUMBER") >= 1
            && player.getNumberResourcesType("GRAIN") >= 1
            && player.getNumberResourcesType("WOOL") >= 1
            && player.getNumbSettlements() < 5;
    }

    private static boolean canBuildCity(Player player) {
        return player.getNumberResourcesType("GRAIN") >= 2
            && player.getNumberResourcesType("ORE") >= 3
            && player.getNumbCities() < 4;
    }

    private static boolean canBuyDevCard(Player player) {
        return player.getNumberResourcesType("GRAIN") >= 1
            && player.getNumberResourcesType("WOOL") >= 1
            && player.getNumberResourcesType("ORE") >= 1;
    }

    private static boolean canDoBankTrade(Player player) {
        for (String resource : RES_TYPES) {
            if (player.getNumberResourcesType(resource) >= 4) {
                return true;
            }
        }
        return false;
    }

    private static boolean performBankTrade(Game game, Player player) {
        String fromResource = mostAbundantTradeableResource(player);
        if (fromResource == null) {
            return false;
        }

        String toResource = chooseTradeTargetResource(player);
        if (toResource == null || toResource.equals(fromResource)) {
            return false;
        }

        ArrayList<String> fromA = new ArrayList<String>();
        for (int i = 0; i < 4; i++) {
            fromA.add(fromResource);
        }

        int code = game.npcTrade(player, toResource, fromA);
        return code == 0;
    }

    private static String mostAbundantTradeableResource(Player player) {
        String best = null;
        int maxCount = -1;
        for (String resource : RES_TYPES) {
            int count = player.getNumberResourcesType(resource);
            if (count >= 4 && count > maxCount) {
                maxCount = count;
                best = resource;
            }
        }
        return best;
    }

    private static String chooseTradeTargetResource(Player player) {
        if (player.getNumberResourcesType("BRICK") == 0) {
            return "BRICK";
        }
        if (player.getNumberResourcesType("LUMBER") == 0) {
            return "LUMBER";
        }
        if (player.getNumberResourcesType("GRAIN") < 2) {
            return "GRAIN";
        }
        if (player.getNumberResourcesType("WOOL") == 0) {
            return "WOOL";
        }
        if (player.getNumberResourcesType("ORE") < 3) {
            return "ORE";
        }
        return "GRAIN";
    }

    private static boolean placeRandomRoad(Game game, Player player, Random rng) {
        Board board = game.getBoard();
        List<EdgeLocation> edges = allEdgeLocations();
        ArrayList<EdgeLocation> legalEdges = new ArrayList<EdgeLocation>();
        int bestScore = Integer.MIN_VALUE;

        for (EdgeLocation edge : edges) {
            if (canPlaceRoadAt(board, player, edge)) {
                int score = roadExpansionScore(board, edge);
                if (score > bestScore) {
                    legalEdges.clear();
                    legalEdges.add(edge);
                    bestScore = score;
                } else if (score == bestScore) {
                    legalEdges.add(edge);
                }
            }
        }

        if (legalEdges.isEmpty()) {
            return false;
        }

        EdgeLocation chosen = legalEdges.get(rng.nextInt(legalEdges.size()));
        return game.placeRoad(player, chosen);
    }

    private static boolean hasAnyLegalRoadPlacement(Game game, Player player) {
        Board board = game.getBoard();
        for (EdgeLocation edge : allEdgeLocations()) {
            if (canPlaceRoadAt(board, player, edge)) {
                return true;
            }
        }
        return false;
    }

    private static int bestRoadExpansionScore(Game game, Player player) {
        Board board = game.getBoard();
        int best = 0;
        for (EdgeLocation edge : allEdgeLocations()) {
            if (canPlaceRoadAt(board, player, edge)) {
                best = Math.max(best, roadExpansionScore(board, edge));
            }
        }
        return best;
    }

    private static int roadExpansionScore(Board board, EdgeLocation edge) {
        int score = 0;
        VertexLocation[] endpoints = endpointsForEdge(edge);
        for (VertexLocation vertex : endpoints) {
            if (vertex == null) {
                continue;
            }
            if (ownerAtVertex(board, vertex) == null && isOpenSettlementSpot(board, vertex)) {
                score += 4;
            }
        }
        return score;
    }

    private static boolean hasAnyLegalSettlementPlacement(Game game, Player player) {
        Board board = game.getBoard();
        for (VertexLocation vertex : allVertexLocations()) {
            if (canPlaceSettlementAt(board, player, vertex)) {
                return true;
            }
        }
        return false;
    }

    private static boolean hasAnyLegalCityPlacement(Game game, Player player) {
        Board board = game.getBoard();
        for (VertexLocation vertex : allVertexLocations()) {
            if (ownerAtVertex(board, vertex) != null
                && ownerAtVertex(board, vertex).equals(player)
                && board.getStructure(vertex).getType() == 0) {
                return true;
            }
        }
        return false;
    }

    private static boolean canPlaceRoadAt(Board board, Player player, EdgeLocation loc) {
        if (board.getRoad(loc).getOwner() != null) {
            return false;
        }

        int x = loc.getXCoord();
        int y = loc.getYCoord();
        int o = loc.getOrientation();

        if (o == 0) {
            return player.equals(ownerAtVertex(board, new VertexLocation(x, y + 1, 1)))
                || player.equals(ownerAtVertex(board, new VertexLocation(x, y, 0)))
                || player.equals(ownerAtRoad(board, new EdgeLocation(x - 1, y, 1)))
                || player.equals(ownerAtRoad(board, new EdgeLocation(x - 1, y, 2)))
                || player.equals(ownerAtRoad(board, new EdgeLocation(x, y + 1, 2)))
                || player.equals(ownerAtRoad(board, new EdgeLocation(x, y, 1)));
        }

        if (o == 1) {
            return player.equals(ownerAtVertex(board, new VertexLocation(x, y, 0)))
                || player.equals(ownerAtVertex(board, new VertexLocation(x + 1, y + 1, 1)))
                || player.equals(ownerAtRoad(board, new EdgeLocation(x, y, 0)))
                || player.equals(ownerAtRoad(board, new EdgeLocation(x, y + 1, 2)))
                || player.equals(ownerAtRoad(board, new EdgeLocation(x, y, 2)))
                || player.equals(ownerAtRoad(board, new EdgeLocation(x + 1, y, 0)));
        }

        return player.equals(ownerAtVertex(board, new VertexLocation(x, y - 1, 0)))
            || player.equals(ownerAtVertex(board, new VertexLocation(x + 1, y + 1, 1)))
            || player.equals(ownerAtRoad(board, new EdgeLocation(x, y, 1)))
            || player.equals(ownerAtRoad(board, new EdgeLocation(x + 1, y, 0)))
            || player.equals(ownerAtRoad(board, new EdgeLocation(x, y - 1, 0)))
            || player.equals(ownerAtRoad(board, new EdgeLocation(x, y - 1, 1)));
    }

    private static boolean canPlaceSettlementAt(Board board, Player player, VertexLocation loc) {
        if (ownerAtVertex(board, loc) != null) {
            return false;
        }

        int x = loc.getXCoord();
        int y = loc.getYCoord();
        int o = loc.getOrientation();

        if (o == 0) {
            boolean connected = player.equals(ownerAtRoad(board, new EdgeLocation(x, y, 0)))
                || player.equals(ownerAtRoad(board, new EdgeLocation(x, y, 1)))
                || player.equals(ownerAtRoad(board, new EdgeLocation(x, y + 1, 2)));
            boolean spacing = ownerAtVertex(board, new VertexLocation(x, y + 1, 1)) == null
                && ownerAtVertex(board, new VertexLocation(x + 1, y + 1, 1)) == null
                && !(y + 2 <= 6 && ownerAtVertex(board, new VertexLocation(x + 1, y + 2, 1)) != null);
            return connected && spacing;
        }

        boolean connected = player.equals(ownerAtRoad(board, new EdgeLocation(x, y - 1, 0)))
            || player.equals(ownerAtRoad(board, new EdgeLocation(x - 1, y - 1, 1)))
            || player.equals(ownerAtRoad(board, new EdgeLocation(x - 1, y - 1, 2)));
        boolean spacing = ownerAtVertex(board, new VertexLocation(x, y - 1, 0)) == null
            && ownerAtVertex(board, new VertexLocation(x - 1, y - 1, 0)) == null
            && !(y - 2 >= 0 && ownerAtVertex(board, new VertexLocation(x - 1, y - 2, 0)) != null);
        return connected && spacing;
    }

    private static boolean isOpenSettlementSpot(Board board, VertexLocation loc) {
        if (ownerAtVertex(board, loc) != null) {
            return false;
        }
        int x = loc.getXCoord();
        int y = loc.getYCoord();
        int o = loc.getOrientation();
        if (o == 0) {
            return ownerAtVertex(board, new VertexLocation(x, y + 1, 1)) == null
                && ownerAtVertex(board, new VertexLocation(x + 1, y + 1, 1)) == null
                && !(y + 2 <= 6 && ownerAtVertex(board, new VertexLocation(x + 1, y + 2, 1)) != null);
        }
        return ownerAtVertex(board, new VertexLocation(x, y - 1, 0)) == null
            && ownerAtVertex(board, new VertexLocation(x - 1, y - 1, 0)) == null
            && !(y - 2 >= 0 && ownerAtVertex(board, new VertexLocation(x - 1, y - 2, 0)) != null);
    }

    private static VertexLocation[] endpointsForEdge(EdgeLocation edge) {
        int x = edge.getXCoord();
        int y = edge.getYCoord();
        int o = edge.getOrientation();
        if (o == 0) {
            return new VertexLocation[] {new VertexLocation(x, y, 0), new VertexLocation(x, y + 1, 1)};
        }
        if (o == 1) {
            return new VertexLocation[] {new VertexLocation(x, y, 0), new VertexLocation(x + 1, y + 1, 1)};
        }
        return new VertexLocation[] {new VertexLocation(x, y - 1, 0), new VertexLocation(x + 1, y + 1, 1)};
    }

    private static Player ownerAtVertex(Board board, VertexLocation loc) {
        if (loc.getXCoord() < 0 || loc.getXCoord() > 6 || loc.getYCoord() < 0 || loc.getYCoord() > 6) {
            return null;
        }
        return board.getStructure(loc).getOwner();
    }

    private static Player ownerAtRoad(Board board, EdgeLocation loc) {
        if (loc.getXCoord() < 0 || loc.getXCoord() > 6 || loc.getYCoord() < 0 || loc.getYCoord() > 6) {
            return null;
        }
        return board.getRoad(loc).getOwner();
    }

    private static boolean placeRandomSettlement(Game game, Player player, Random rng) {
        List<VertexLocation> vertices = allVertexLocations();
        Collections.shuffle(vertices, rng);
        for (VertexLocation vertex : vertices) {
            if (game.placeSettlement(player, vertex)) {
                return true;
            }
        }
        return false;
    }

    private static boolean placeRandomCity(Game game, Player player, Random rng) {
        List<VertexLocation> vertices = allVertexLocations();
        Collections.shuffle(vertices, rng);
        for (VertexLocation vertex : vertices) {
            if (game.getBoard().getStructure(vertex).getOwner() != null
                && game.getBoard().getStructure(vertex).getOwner().equals(player)
                && game.getBoard().getStructure(vertex).getType() == 0
                && game.placeCity(player, vertex)) {
                return true;
            }
        }
        return false;
    }

    private static void maybeMoveRobber(Game game, Random rng) {
        Board board = game.getBoard();
        List<Location> tileLocations = allTileLocations(board);
        Collections.shuffle(tileLocations, rng);
        for (Location loc : tileLocations) {
            if (board.moveRobber(loc)) {
                return;
            }
        }
    }

    private static void safeHalfCards(ArrayList<Player> players, Random rng) {
        for (Player player : players) {
            int total = player.getNumberResourcesType("BRICK")
                + player.getNumberResourcesType("WOOL")
                + player.getNumberResourcesType("ORE")
                + player.getNumberResourcesType("GRAIN")
                + player.getNumberResourcesType("LUMBER");

            if (total <= 7) {
                continue;
            }

            int toDiscard = total / 2;
            for (int i = 0; i < toDiscard; i++) {
                ArrayList<String> owned = player.getOwnedResources();
                if (owned.isEmpty()) {
                    break;
                }
                String pick = owned.get(rng.nextInt(owned.size()));
                int current = player.getNumberResourcesType(pick);
                if (current > 0) {
                    player.setNumberResourcesType(pick, current - 1);
                }
            }
        }
    }

    private static List<Location> allTileLocations(Board board) {
        ArrayList<Location> out = new ArrayList<>();
        for (int x = 1; x <= 5; x++) {
            for (int y = 1; y <= 5; y++) {
                if (board.getTiles()[x][y].getType() != null) {
                    out.add(new Location(x, y));
                }
            }
        }
        return out;
    }

    private static List<VertexLocation> allVertexLocations() {
        ArrayList<VertexLocation> out = new ArrayList<>();
        for (int x = 1; x <= 5; x++) {
            for (int y = 1; y <= 5; y++) {
                for (int o = 0; o < 2; o++) {
                    out.add(new VertexLocation(x, y, o));
                }
            }
        }
        return out;
    }

    private static List<EdgeLocation> allEdgeLocations() {
        ArrayList<EdgeLocation> out = new ArrayList<>();
        for (int x = 1; x <= 5; x++) {
            for (int y = 1; y <= 5; y++) {
                for (int o = 0; o < 3; o++) {
                    out.add(new EdgeLocation(x, y, o));
                }
            }
        }
        return out;
    }

    private static String snapshotJsonLine(
        Game game,
        ArrayList<Player> players,
        String gameId,
        int turnIndex,
        int currentPlayerId,
        String currentPlayerName,
        String actionTaken,
        List<String> recentActions,
        int roll
    ) {
        Board board = game.getBoard();
        Location robberLoc = board.getRobberLocation();

        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"game_id\":\"").append(escape(gameId)).append("\",");
        sb.append("\"position_id\":\"").append(escape(gameId)).append("_").append(turnIndex).append("\",");
        sb.append("\"row_index_in_game\":").append(turnIndex).append(",");
        sb.append("\"turn_metadata\":{");
        sb.append("\"turn_index\":").append(turnIndex).append(",");
        sb.append("\"current_player_id\":\"").append(currentPlayerId).append("\",");
        sb.append("\"current_player_name\":\"").append(escape(currentPlayerName)).append("\",");
        sb.append("\"current_phase\":\"main\"");
        sb.append("},");
        sb.append("\"action_taken\":\"").append(escape(actionTaken)).append("\",");
        sb.append("\"roll\":").append(roll).append(",");
        sb.append("\"player_state\":[");
        for (int i = 0; i < players.size(); i++) {
            Player p = players.get(i);
            sb.append("{");
            sb.append("\"player_name\":\"").append(escape(p.getName())).append("\",");
            sb.append("\"victory_points\":").append(p.getVictoryPoints()).append(",");
            sb.append("\"resources\":{");
            sb.append("\"BRICK\":").append(p.getNumberResourcesType("BRICK")).append(",");
            sb.append("\"WOOL\":").append(p.getNumberResourcesType("WOOL")).append(",");
            sb.append("\"ORE\":").append(p.getNumberResourcesType("ORE")).append(",");
            sb.append("\"GRAIN\":").append(p.getNumberResourcesType("GRAIN")).append(",");
            sb.append("\"LUMBER\":").append(p.getNumberResourcesType("LUMBER"));
            sb.append("}");
            sb.append("}");
            if (i < players.size() - 1) {
                sb.append(",");
            }
        }
        sb.append("],");

        sb.append("\"snapshot\":{");
        sb.append("\"turn_index\":").append(turnIndex).append(",");
        sb.append("\"current_phase\":\"main\",");
        sb.append("\"current_player_id\":\"").append(currentPlayerId).append("\",");
        sb.append("\"current_player_name\":\"").append(escape(currentPlayerName)).append("\",");
        sb.append("\"last_action\":\"").append(escape(actionTaken)).append("\",");
        sb.append("\"latest_dice_roll\":").append(roll).append(",");

        sb.append("\"recent_actions\":[");
        for (int i = 0; i < recentActions.size(); i++) {
            sb.append("\"").append(escape(recentActions.get(i))).append("\"");
            if (i < recentActions.size() - 1) {
                sb.append(",");
            }
        }
        sb.append("],");

        if (robberLoc == null) {
            sb.append("\"robber\":null,");
        } else {
            sb.append("\"robber\":{");
            sb.append("\"x\":").append(robberLoc.getXCoord()).append(",");
            sb.append("\"y\":").append(robberLoc.getYCoord());
            sb.append("},");
        }

        sb.append("\"tiles\":[");
        boolean firstTile = true;
        Tile[][] tiles = board.getTiles();
        for (int x = 0; x <= 6; x++) {
            for (int y = 0; y <= 6; y++) {
                Tile tile = tiles[x][y];
                if (tile == null || tile.getType() == null) {
                    continue;
                }
                if (!firstTile) {
                    sb.append(",");
                }
                firstTile = false;
                sb.append("{");
                sb.append("\"x\":").append(x).append(",");
                sb.append("\"y\":").append(y).append(",");
                sb.append("\"type\":\"").append(escape(tile.getType())).append("\",");
                sb.append("\"number\":").append(tile.getNumber()).append(",");
                sb.append("\"robber\":").append(tile.hasRobber());
                sb.append("}");
            }
        }
        sb.append("],");

        sb.append("\"structures\":[");
        boolean firstStructure = true;
        Structure[][][] structures = board.getStructures();
        for (int x = 0; x <= 6; x++) {
            for (int y = 0; y <= 6; y++) {
                for (int o = 0; o <= 1; o++) {
                    Structure structure = structures[x][y][o];
                    if (structure == null || structure.getOwner() == null) {
                        continue;
                    }
                    if (!firstStructure) {
                        sb.append(",");
                    }
                    firstStructure = false;
                    String ownerName = structure.getOwner().getName();
                    sb.append("{");
                    sb.append("\"x\":").append(x).append(",");
                    sb.append("\"y\":").append(y).append(",");
                    sb.append("\"orientation\":").append(o).append(",");
                    sb.append("\"type\":").append(structure.getType()).append(",");
                    sb.append("\"owner_id\":\"name:").append(escape(ownerName)).append("\",");
                    sb.append("\"owner_name\":\"").append(escape(ownerName)).append("\"");
                    sb.append("}");
                }
            }
        }
        sb.append("],");

        sb.append("\"roads\":[");
        boolean firstRoad = true;
        Road[][][] roads = board.getRoads();
        for (int x = 0; x <= 6; x++) {
            for (int y = 0; y <= 6; y++) {
                for (int o = 0; o <= 2; o++) {
                    Road road = roads[x][y][o];
                    if (road == null || road.getOwner() == null) {
                        continue;
                    }
                    if (!firstRoad) {
                        sb.append(",");
                    }
                    firstRoad = false;
                    String ownerName = road.getOwner().getName();
                    sb.append("{");
                    sb.append("\"x\":").append(x).append(",");
                    sb.append("\"y\":").append(y).append(",");
                    sb.append("\"orientation\":").append(o).append(",");
                    sb.append("\"owner_id\":\"name:").append(escape(ownerName)).append("\",");
                    sb.append("\"owner_name\":\"").append(escape(ownerName)).append("\"");
                    sb.append("}");
                }
            }
        }
        sb.append("],");

        sb.append("\"players\":[");
        for (int i = 0; i < players.size(); i++) {
            Player p = players.get(i);
            int brick = p.getNumberResourcesType("BRICK");
            int wool = p.getNumberResourcesType("WOOL");
            int ore = p.getNumberResourcesType("ORE");
            int grain = p.getNumberResourcesType("GRAIN");
            int lumber = p.getNumberResourcesType("LUMBER");
            int total = brick + wool + ore + grain + lumber;
            int roadsPlaced = p.getNumbRoads();
            int settlementsPlaced = p.getNumbSettlements();
            int citiesPlaced = p.getNumbCities();

            sb.append("{");
            sb.append("\"player_id\":\"name:").append(escape(p.getName())).append("\",");
            sb.append("\"player_name\":\"").append(escape(p.getName())).append("\",");
            sb.append("\"victory_points\":").append(p.getVictoryPoints()).append(",");
            sb.append("\"victory_points_visible\":").append(p.getVictoryPoints()).append(",");
            sb.append("\"knights_played_visible\":").append(p.getNumbKnights()).append(",");
            sb.append("\"largest_army_flag\":").append(p.hasLargestArmy()).append(",");
            sb.append("\"resource_cards\":{");
            sb.append("\"BRICK\":").append(brick).append(",");
            sb.append("\"WOOL\":").append(wool).append(",");
            sb.append("\"ORE\":").append(ore).append(",");
            sb.append("\"GRAIN\":").append(grain).append(",");
            sb.append("\"LUMBER\":").append(lumber);
            sb.append("},");
            sb.append("\"total_resource_cards\":").append(total).append(",");
            sb.append("\"ports\":[");
            boolean[] ports = p.getPorts();
            for (int portIndex = 0; portIndex < ports.length; portIndex++) {
                sb.append(ports[portIndex]);
                if (portIndex < ports.length - 1) {
                    sb.append(",");
                }
            }
            sb.append("],");
            sb.append("\"remaining_pieces\":{");
            sb.append("\"roads_remaining\":").append(15 - roadsPlaced).append(",");
            sb.append("\"settlements_remaining\":").append(5 - settlementsPlaced).append(",");
            sb.append("\"cities_remaining\":").append(4 - citiesPlaced).append(",");
            sb.append("\"roads_placed_reported\":").append(roadsPlaced).append(",");
            sb.append("\"settlements_placed_reported\":").append(settlementsPlaced).append(",");
            sb.append("\"cities_placed_reported\":").append(citiesPlaced);
            sb.append("}");
            sb.append("}");
            if (i < players.size() - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        sb.append("}");
        sb.append("}");
        return sb.toString();
    }

    private static String escape(String value) {
        if (value == null) {
            return "";
        }
        return value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static int[] resourceVector(Player player) {
        return new int[] {
            player.getNumberResourcesType("BRICK"),
            player.getNumberResourcesType("WOOL"),
            player.getNumberResourcesType("ORE"),
            player.getNumberResourcesType("GRAIN"),
            player.getNumberResourcesType("LUMBER")
        };
    }

    private static void assertResourceVectorEquals(int[] expected, int[] actual, String message) {
        for (int i = 0; i < expected.length; i++) {
            if (expected[i] != actual[i]) {
                throw new IllegalStateException(message + " expected=" + Arrays.toString(expected) + " actual=" + Arrays.toString(actual));
            }
        }
    }

    private static void assertResourceDelta(int[] after, int[] before, int[] expectedDelta, String actionName) {
        for (int i = 0; i < before.length; i++) {
            int delta = after[i] - before[i];
            if (delta != expectedDelta[i]) {
                throw new IllegalStateException("resource_delta_mismatch action=" + actionName
                    + " res=" + RES_TYPES[i]
                    + " before=" + before[i]
                    + " after=" + after[i]
                    + " delta=" + delta
                    + " expected=" + expectedDelta[i]);
            }
        }
    }

    private static int totalResources(ArrayList<Player> players) {
        int sum = 0;
        for (Player p : players) {
            sum += p.getNumberResourcesType("BRICK");
            sum += p.getNumberResourcesType("WOOL");
            sum += p.getNumberResourcesType("ORE");
            sum += p.getNumberResourcesType("GRAIN");
            sum += p.getNumberResourcesType("LUMBER");
        }
        return sum;
    }

    private static int totalVictoryPoints(ArrayList<Player> players) {
        int sum = 0;
        for (Player p : players) {
            sum += p.getVictoryPoints();
        }
        return sum;
    }

    private static String stateFingerprint(ArrayList<Player> players) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < players.size(); i++) {
            Player p = players.get(i);
            sb.append(p.getName()).append(":");
            sb.append(p.getVictoryPoints()).append("|");
            sb.append(p.getNumberResourcesType("BRICK")).append(",");
            sb.append(p.getNumberResourcesType("WOOL")).append(",");
            sb.append(p.getNumberResourcesType("ORE")).append(",");
            sb.append(p.getNumberResourcesType("GRAIN")).append(",");
            sb.append(p.getNumberResourcesType("LUMBER")).append("|");
            sb.append(p.getNumbRoads()).append(",");
            sb.append(p.getNumbSettlements()).append(",");
            sb.append(p.getNumbCities()).append(";");
        }
        return sb.toString();
    }

    private static void assertAllResourcesNonNegative(ArrayList<Player> players, String context) {
        for (Player p : players) {
            for (String res : RES_TYPES) {
                if (p.getNumberResourcesType(res) < 0) {
                    throw new IllegalStateException("negative_resource context=" + context + " player=" + p.getName() + " res=" + res + " value=" + p.getNumberResourcesType(res));
                }
            }
            if (p.getVictoryPoints() < 0) {
                throw new IllegalStateException("negative_vp context=" + context + " player=" + p.getName() + " vp=" + p.getVictoryPoints());
            }
            if (p.getVictoryPoints() > 15) {
                throw new IllegalStateException("vp_unreasonably_high context=" + context + " player=" + p.getName() + " vp=" + p.getVictoryPoints());
            }
        }
    }

    private static boolean isPassOnly(ArrayList<String> actions) {
        if (actions.isEmpty()) {
            return true;
        }
        for (String action : actions) {
            if (!"pass".equals(action)) {
                return false;
            }
        }
        return true;
    }

    private static void fail(Config config, String code, String detail) {
        String message = code + " detail=" + detail;
        if (config.debug) {
            System.err.println("[INVARIANT_FAIL] " + message);
        }
        throw new IllegalStateException(message);
    }

    private static String resourceVectorString(int[] vector) {
        return Arrays.toString(vector);
    }

    private static void logAttempt(
        Config config,
        String gameId,
        int turnIndex,
        int actionIndex,
        Player player,
        String action,
        boolean success,
        String reason,
        int[] before,
        int[] after
    ) {
        if (!config.debug) {
            return;
        }
        System.err.println("[ACTION] game_id=" + gameId
            + " turn=" + turnIndex
            + " action_index=" + actionIndex
            + " player=" + player.getName()
            + " action=" + action
            + " success=" + success
            + " reason=" + reason
            + " before=" + resourceVectorString(before)
            + " after=" + resourceVectorString(after));
    }
}
