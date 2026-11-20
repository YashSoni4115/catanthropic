
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
        String openingPolicy = "random";
        String openingModelDir = "";
        String openingPythonExec = "python3";
        int openingSampleTopK = 0;
        // External-policy mode: on the AI player's turn, block for a decision
        // over stdin instead of running the built-in policy.
        boolean externalPolicy = false;
        int aiPlayerIndex = -1;
    }

    /**
     * Client for the interactive Python bridge (v1).
     * Communicates over Java's own System.in / System.out using
     * line-delimited JSON.  See ai/bridge/game_bridge_v1.py for the
     * Python counterpart and the protocol spec.
     */
    private static final class ExternalPolicyClient {
        private final BufferedReader in;
        private int nextRequestId = 1;
        int fallbackCount = 0;

        ExternalPolicyClient() {
            this.in = new BufferedReader(new InputStreamReader(System.in));
        }

        /** Send a JSON line to Python and read its JSON response line. */
        String exchange(String requestJson) throws IOException {
            System.out.println(requestJson);
            System.out.flush();
            String line = in.readLine();
            if (line == null) {
                throw new IOException("external policy: EOF from Python bridge");
            }
            return line;
        }

        int nextId() { return nextRequestId++; }
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

    private static final class OpeningScorer {
        private final Process process;
        private final BufferedWriter input;
        private final BufferedReader output;

        OpeningScorer(Config config) throws IOException {
            ArrayList<String> command = new ArrayList<String>();
            command.add(config.openingPythonExec);
            command.add("-u");
            command.add("ai/selfplay/opening_scorer_v1.py");
            command.add("--model-dir");
            command.add(config.openingModelDir);
            if (config.openingSampleTopK > 0) {
                command.add("--sample-topk");
                command.add(String.valueOf(config.openingSampleTopK));
            }

            ProcessBuilder pb = new ProcessBuilder(command);
            pb.redirectError(ProcessBuilder.Redirect.INHERIT);
            this.process = pb.start();
            this.input = new BufferedWriter(new OutputStreamWriter(process.getOutputStream()));
            this.output = new BufferedReader(new InputStreamReader(process.getInputStream()));
        }

        /**
         * Send current board snapshot, receive ranked (settlement, road) placements.
         * Returns list of int[6] arrays: {sx, sy, so, rx, ry, ro}.
         */
        List<int[]> scorePlacements(String snapshotJson) throws IOException {
            input.write(snapshotJson);
            input.newLine();
            input.flush();

            String response = output.readLine();
            if (response == null) {
                throw new IOException("opening scorer returned EOF");
            }
            return parsePlacements(response);
        }

        void close() {
            try { input.close(); } catch (Exception ignored) {}
            try { output.close(); } catch (Exception ignored) {}
            try { process.destroy(); } catch (Exception ignored) {}
        }
    }

    private static List<int[]> parsePlacements(String json) {
        List<int[]> result = new ArrayList<int[]>();
        // Minimal JSON parsing for {"placements":[{"sx":...,"sy":...,"so":...,"rx":...,"ry":...,"ro":...}]}
        int idx = json.indexOf("\"placements\"");
        if (idx < 0) return result;
        int arrStart = json.indexOf('[', idx);
        if (arrStart < 0) return result;
        int arrEnd = json.indexOf(']', arrStart);
        if (arrEnd < 0) return result;
        String arrContent = json.substring(arrStart + 1, arrEnd);
        // Split on "}," to get individual placement objects
        String[] parts = arrContent.split("\\},");
        for (String part : parts) {
            part = part.trim();
            if (part.isEmpty()) continue;
            if (!part.endsWith("}")) part = part + "}";
            int sx = extractInt(part, "\"sx\"");
            int sy = extractInt(part, "\"sy\"");
            int so = extractInt(part, "\"so\"");
            int rx = extractInt(part, "\"rx\"");
            int ry = extractInt(part, "\"ry\"");
            int ro = extractInt(part, "\"ro\"");
            result.add(new int[]{sx, sy, so, rx, ry, ro});
        }
        return result;
    }

    private static int extractInt(String json, String key) {
        int idx = json.indexOf(key);
        if (idx < 0) return -1;
        int colon = json.indexOf(':', idx + key.length());
        if (colon < 0) return -1;
        StringBuilder sb = new StringBuilder();
        for (int i = colon + 1; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '-' || (c >= '0' && c <= '9')) {
                sb.append(c);
            } else if (sb.length() > 0) {
                break;
            }
        }
        if (sb.length() == 0) return -1;
        return Integer.parseInt(sb.toString());
    }

    private static final class ActionTrace {
        ArrayList<String> successfulActions = new ArrayList<String>();
    }

    private static final class SnapshotCursor {
        private int rowIndex = 0;

        int next() {
            return rowIndex++;
        }
    }

    public static void main(String[] args) {
        Config config = parseArgs(args);
        String gameId = UUID.randomUUID().toString();
        Random rng = config.hasSeed ? new Random(config.seed) : new Random();
        ArrayList<Player> players = makePlayers();
        Game game = new Game(players);
        List<String> trajectory = new ArrayList<>();
        SnapshotCursor snapshotCursor = new SnapshotCursor();
        int turnIndex = 0;
        int consecutivePassTurns = 0;
        int noProgressTurns = 0;
        HashSet<Integer> seenTurns = new HashSet<Integer>();
        String terminationReason = "error";
        GuidedModelScorer scorer = null;
        OpeningScorer openingScorer = null;
        ExternalPolicyClient externalClient = null;

        try {
            if (config.externalPolicy) {
                if (config.aiPlayerIndex < 0) {
                    throw new IllegalArgumentException("--external-policy requires --ai-player <index>");
                }
                externalClient = new ExternalPolicyClient();
            }
            if (config.guided) {
                if (config.guidedModelPath == null || config.guidedModelPath.trim().isEmpty()) {
                    throw new IllegalArgumentException("--guided requires --guided-model-path");
                }
                scorer = new GuidedModelScorer(config);
            }

            if ("value_model".equals(config.openingPolicy) || "value_model_topk".equals(config.openingPolicy)) {
                if (config.openingModelDir == null || config.openingModelDir.trim().isEmpty()) {
                    throw new IllegalArgumentException("--opening-policy value_model requires --opening-model-dir");
                }
                openingScorer = new OpeningScorer(config);
            }

            Player setupStarter = players.get(0);
            trajectory.add(
                snapshotJsonLine(
                    game,
                    players,
                    gameId,
                    snapshotCursor.next(),
                    -1,
                    0,
                    setupStarter.getName(),
                    "setup_pre_initial_placements",
                    Collections.<String>emptyList(),
                    null,
                    "setup",
                    "setup_pre_initial_placements",
                    0
                )
            );

            runInitialPlacements(game, players, rng, trajectory, gameId, snapshotCursor, openingScorer, config, externalClient);
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
                    boolean isAi = externalClient != null && currentPlayerId == config.aiPlayerIndex;
                    if (isAi) {
                        maybeMoveRobberExternal(game, players, current, rng, externalClient, config, turnIndex, gameId);
                    } else {
                        maybeMoveRobber(game, rng);
                    }
                }
                if (rollApplications != 1) {
                    fail(config, "roll_applied_invalid_count", "count=" + rollApplications + " turn=" + turnIndex);
                }

                int totalResAfterRoll = totalResources(players);

                ActionTrace trace;
                boolean isAiTurn = externalClient != null && currentPlayerId == config.aiPlayerIndex;
                if (isAiTurn) {
                    trace = performExternalTurnActions(game, players, current, rng, config, gameId, turnIndex, roll, externalClient);
                } else {
                    trace = performRandomTurnActions(game, players, current, rng, config, gameId, turnIndex, roll, scorer);
                }
                ArrayList<String> actions = trace.successfulActions;

                String actionLabel = actions.isEmpty() ? "pass" : String.join("+", actions);
                trajectory.add(
                    snapshotJsonLine(
                        game,
                        players,
                        gameId,
                        snapshotCursor.next(),
                        turnIndex,
                        currentPlayerId,
                        current.getName(),
                        actionLabel,
                        actions,
                        Integer.valueOf(roll),
                        "main",
                        actionLabel,
                        -1
                    )
                );

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

            // Bridge marker (before the trajectory dump) so the Python
            // bridge can end its read loop cleanly.
            if (externalClient != null) {
                boolean aiWon = false;
                String aiName = "";
                if (config.aiPlayerIndex >= 0 && config.aiPlayerIndex < players.size()) {
                    aiName = players.get(config.aiPlayerIndex).getName();
                    aiWon = "winner".equals(terminationReason) && winner != null
                            && aiName.equals(winner.getName());
                }
                System.out.println(
                    "{\"type\":\"game_over\",\"winner\":\"" + escape(winnerName)
                    + "\",\"turns\":" + turnIndex
                    + ",\"ai_player\":\"" + escape(aiName)
                    + "\",\"ai_won\":" + aiWon
                    + ",\"termination_reason\":\"" + escape(terminationReason)
                    + "\",\"fallbacks\":" + externalClient.fallbackCount + "}"
                );
                System.out.flush();
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
            if (openingScorer != null) {
                openingScorer.close();
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
            } else if ("--opening-policy".equals(arg) && i + 1 < args.length) {
                config.openingPolicy = args[++i];
            } else if ("--opening-model-dir".equals(arg) && i + 1 < args.length) {
                config.openingModelDir = args[++i];
            } else if ("--opening-python-exec".equals(arg) && i + 1 < args.length) {
                config.openingPythonExec = args[++i];
            } else if ("--opening-sample-topk".equals(arg) && i + 1 < args.length) {
                config.openingSampleTopK = Integer.parseInt(args[++i]);
            } else if ("--external-policy".equals(arg)) {
                config.externalPolicy = true;
            } else if ("--ai-player".equals(arg) && i + 1 < args.length) {
                config.aiPlayerIndex = Integer.parseInt(args[++i]);
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

    private static void runInitialPlacements(
        Game game,
        ArrayList<Player> players,
        Random rng,
        List<String> trajectory,
        String gameId,
        SnapshotCursor snapshotCursor,
        OpeningScorer openingScorer,
        Config config,
        ExternalPolicyClient externalClient
    ) {
        int setupStepIndex = 1;
        for (Player player : players) {
            boolean isAi = externalClient != null && players.indexOf(player) == config.aiPlayerIndex;
            if (isAi) {
                placeInitialSettlementAndRoadExternal(game, players, player, rng, externalClient, config, gameId);
            } else {
                placeInitialSettlementAndRoad(game, players, player, rng, openingScorer, config, gameId);
            }
            trajectory.add(
                snapshotJsonLine(
                    game,
                    players,
                    gameId,
                    snapshotCursor.next(),
                    -1,
                    players.indexOf(player),
                    player.getName(),
                    "setup_initial_placement",
                    Arrays.asList("place_initial_settlement", "place_initial_road"),
                    null,
                    "setup",
                    "post_initial_placement",
                    setupStepIndex++
                )
            );
        }
        for (int i = players.size() - 1; i >= 0; i--) {
            Player player = players.get(i);
            boolean isAi = externalClient != null && i == config.aiPlayerIndex;
            if (isAi) {
                placeInitialSettlementAndRoadExternal(game, players, player, rng, externalClient, config, gameId);
            } else {
                placeInitialSettlementAndRoad(game, players, player, rng, openingScorer, config, gameId);
            }
            trajectory.add(
                snapshotJsonLine(
                    game,
                    players,
                    gameId,
                    snapshotCursor.next(),
                    -1,
                    i,
                    player.getName(),
                    "setup_initial_placement",
                    Arrays.asList("place_initial_settlement", "place_initial_road"),
                    null,
                    "setup",
                    "post_initial_placement",
                    setupStepIndex++
                )
            );
        }
    }

    private static void placeInitialSettlementAndRoad(
        Game game,
        ArrayList<Player> players,
        Player player,
        Random rng,
        OpeningScorer openingScorer,
        Config config,
        String gameId
    ) {
        if (openingScorer != null) {
            try {
                // Build a snapshot of the current board state for the scorer
                String snapshot = snapshotJsonLine(
                    game, players, gameId, -1, -1,
                    players.indexOf(player), player.getName(),
                    "setup_opening_query", Collections.<String>emptyList(),
                    null, "setup", "opening_query", -1
                );
                List<int[]> placements = openingScorer.scorePlacements(snapshot);

                boolean useTopK = "value_model_topk".equals(config.openingPolicy)
                                  && config.openingSampleTopK > 0;

                for (int[] placement : placements) {
                    int sx = placement[0], sy = placement[1], so = placement[2];
                    int rx = placement[3], ry = placement[4], ro = placement[5];

                    // For top-k sampling, pick uniformly from available placements
                    if (useTopK && placements.size() > 1) {
                        int pick = rng.nextInt(Math.min(config.openingSampleTopK, placements.size()));
                        int[] sampled = placements.get(pick);
                        sx = sampled[0]; sy = sampled[1]; so = sampled[2];
                        rx = sampled[3]; ry = sampled[4]; ro = sampled[5];
                    }

                    VertexLocation vLoc = new VertexLocation(sx, sy, so);
                    EdgeLocation eLoc = new EdgeLocation(rx, ry, ro);

                    if (game.getBoard().placeStructureNoRoad(vLoc, player)) {
                        if (game.placeRoad(player, eLoc)) {
                            if (config.debug) {
                                System.err.println("[OPENING] game=" + gameId
                                    + " player=" + player.getName()
                                    + " policy=" + config.openingPolicy
                                    + " settlement=" + sx + ":" + sy + ":" + so
                                    + " road=" + rx + ":" + ry + ":" + ro);
                            }
                            return;
                        }
                        // Road failed — undo by noting this shouldn't happen with valid scorer output
                        if (config.debug) {
                            System.err.println("[OPENING] road placement failed at "
                                + rx + ":" + ry + ":" + ro + ", falling back to random road");
                        }
                        placeRandomRoad(game, player, rng);
                        return;
                    }
                    // Settlement failed, try next candidate (greedy mode) or break (topk already sampled)
                    if (useTopK) break;
                }
                // If all value-model candidates failed, fall back to random
                if (config.debug) {
                    System.err.println("[OPENING] all value_model candidates failed, falling back to random");
                }
            } catch (IOException e) {
                System.err.println("[OPENING] scorer error: " + e.getMessage() + ", falling back to random");
            }
        }
        // Fallback: random placement (original behavior)
        placeRandomInitialSettlementAndRoad(game, player, rng);
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

    /**
     * External-policy variant of initial placement.  Emits
     * setup_settlement then setup_road decision_requests; validates and
     * applies each via the same underlying Board methods.  Falls back
     * to random on any illegal / malformed response.
     */
    private static void placeInitialSettlementAndRoadExternal(
        Game game, ArrayList<Player> players, Player player, Random rng,
        ExternalPolicyClient client, Config config, String gameId
    ) {
        // ---- Settlement ----
        int settlementReqId = client.nextId();
        String settlementReq = buildDecisionRequest(
            settlementReqId, "setup_settlement", players, player, game, gameId,
            -1, 0, Collections.singletonList("setup_settlement")
        );
        VertexLocation placedVertex = null;
        try {
            String response = client.exchange(settlementReq);
            String kind = extractStringField(response, "action_kind");
            if ("setup_settlement".equals(kind)) {
                int vx = extractInt(response, "\"vertex_x\"");
                int vy = extractInt(response, "\"vertex_y\"");
                int vo = extractInt(response, "\"vertex_o\"");
                if (vx >= 0 && vy >= 0 && vo >= 0) {
                    VertexLocation vLoc = new VertexLocation(vx, vy, vo);
                    if (game.getBoard().placeStructureNoRoad(vLoc, player)) {
                        placedVertex = vLoc;
                    } else {
                        emitBridgeError(settlementReqId, "engine_rejected_setup_vertex");
                    }
                } else {
                    emitBridgeError(settlementReqId, "bad_setup_vertex");
                }
            } else {
                emitBridgeError(settlementReqId, "expected_setup_settlement_got:" + kind);
            }
        } catch (IOException e) {
            System.err.println("[BRIDGE] setup settlement I/O error: " + e.getMessage());
        }
        if (placedVertex == null) {
            client.fallbackCount++;
            placeRandomInitialSettlementAndRoad(game, player, rng);
            return;
        }

        // ---- Road ----
        int roadReqId = client.nextId();
        String roadReq = buildDecisionRequest(
            roadReqId, "setup_road", players, player, game, gameId,
            -1, 0, Collections.singletonList("setup_road")
        );
        boolean roadPlaced = false;
        try {
            String response = client.exchange(roadReq);
            String kind = extractStringField(response, "action_kind");
            if ("setup_road".equals(kind)) {
                int ex = extractInt(response, "\"edge_x\"");
                int ey = extractInt(response, "\"edge_y\"");
                int eo = extractInt(response, "\"edge_o\"");
                if (ex >= 0 && ey >= 0 && eo >= 0) {
                    EdgeLocation eLoc = new EdgeLocation(ex, ey, eo);
                    if (game.placeRoad(player, eLoc)) {
                        roadPlaced = true;
                    } else {
                        emitBridgeError(roadReqId, "engine_rejected_setup_edge");
                    }
                } else {
                    emitBridgeError(roadReqId, "bad_setup_edge");
                }
            } else {
                emitBridgeError(roadReqId, "expected_setup_road_got:" + kind);
            }
        } catch (IOException e) {
            System.err.println("[BRIDGE] setup road I/O error: " + e.getMessage());
        }
        if (!roadPlaced) {
            client.fallbackCount++;
            placeRandomRoad(game, player, rng);
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

    /**
     * External-policy variant of {@link #performRandomTurnActions}.
     *
     * On each iteration: emit a decision_request line containing the
     * current snapshot + legal action kinds; block for a decision_response;
     * validate + apply the action via the same underlying Game methods.
     * Illegal / malformed responses log an error line and fall back to
     * the built-in random policy for that iteration (game continues).
     */
    private static ActionTrace performExternalTurnActions(
        Game game,
        ArrayList<Player> players,
        Player player,
        Random rng,
        Config config,
        String gameId,
        int turnIndex,
        int roll,
        ExternalPolicyClient client
    ) {
        ActionTrace trace = new ActionTrace();
        // Cap iterations so a broken Python bridge cannot spin forever.
        int hardCap = 10;
        for (int i = 0; i < hardCap; i++) {
            List<String> candidates = legalActionKinds(game, player);
            if (candidates.isEmpty()) {
                break;
            }
            int reqId = client.nextId();
            String req = buildDecisionRequest(
                reqId, "main", players, player, game, gameId, turnIndex, roll, candidates
            );
            String responseLine;
            try {
                responseLine = client.exchange(req);
            } catch (IOException e) {
                System.err.println("[BRIDGE] request " + reqId + " I/O error: " + e.getMessage());
                client.fallbackCount++;
                return performRandomTurnActions(game, players, player, rng, config, gameId, turnIndex, roll, null);
            }

            String actionType = extractStringField(responseLine, "type");
            if (!"decision_response".equals(actionType)) {
                System.err.println("[BRIDGE] request " + reqId + " unexpected response type: " + actionType);
                emitBridgeError(reqId, "bad_response_type");
                client.fallbackCount++;
                break;
            }
            String kind = extractStringField(responseLine, "action_kind");
            if (kind == null || kind.isEmpty()) {
                emitBridgeError(reqId, "missing_action_kind");
                client.fallbackCount++;
                break;
            }
            if ("end_turn".equals(kind) || "pass".equals(kind)) {
                trace.successfulActions.add("pass");
                break;
            }
            if (!candidates.contains(kind)) {
                emitBridgeError(reqId, "illegal_action_kind:" + kind);
                client.fallbackCount++;
                break;
            }

            boolean success = applyExternalAction(game, player, kind, responseLine, rng, config, gameId, turnIndex, i);
            if (!success) {
                emitBridgeError(reqId, "engine_rejected:" + kind);
                client.fallbackCount++;
                // Try one built-in random action instead of hanging on the same kind.
                String fallback = chooseActionKind(game, players, player, candidates, rng, config, turnIndex, roll, null);
                if (applyActionKind(game, player, fallback, rng, config, gameId, turnIndex, i)) {
                    trace.successfulActions.add(fallback);
                    if ("pass".equals(fallback)) break;
                }
                continue;
            }
            trace.successfulActions.add(kind);
        }
        return trace;
    }

    private static String buildDecisionRequest(
        int reqId, String phase, ArrayList<Player> players, Player current,
        Game game, String gameId, int turnIndex, int roll, List<String> legalKinds
    ) {
        String snap = snapshotJsonLine(
            game, players, gameId, -1, turnIndex,
            players.indexOf(current), current.getName(),
            "decision_request", Collections.<String>emptyList(),
            Integer.valueOf(roll), phase, "decision_request", -1
        );
        StringBuilder sb = new StringBuilder();
        sb.append("{\"type\":\"decision_request\",\"request_id\":").append(reqId);
        sb.append(",\"turn\":").append(turnIndex);
        sb.append(",\"phase\":\"").append(escape(phase)).append("\"");
        sb.append(",\"player\":\"").append(escape(current.getName())).append("\"");
        sb.append(",\"legal_action_kinds\":[");
        for (int i = 0; i < legalKinds.size(); i++) {
            sb.append("\"").append(escape(legalKinds.get(i))).append("\"");
            if (i < legalKinds.size() - 1) sb.append(",");
        }
        sb.append("],\"snapshot_payload\":").append(snap).append("}");
        return sb.toString();
    }

    private static void emitBridgeError(int reqId, String reason) {
        System.out.println(
            "{\"type\":\"error\",\"request_id\":" + reqId
            + ",\"reason\":\"" + escape(reason) + "\"}"
        );
        System.out.flush();
    }

    /** Minimal string-field extractor for the flat protocol JSON. */
    private static String extractStringField(String json, String key) {
        String needle = "\"" + key + "\"";
        int idx = json.indexOf(needle);
        if (idx < 0) return null;
        int colon = json.indexOf(':', idx + needle.length());
        if (colon < 0) return null;
        int quote = json.indexOf('"', colon + 1);
        if (quote < 0) return null;
        int close = json.indexOf('"', quote + 1);
        if (close < 0) return null;
        return json.substring(quote + 1, close);
    }

    /** Apply an externally-chosen action; params are parsed from responseLine. */
    private static boolean applyExternalAction(
        Game game, Player player, String kind, String responseLine,
        Random rng, Config config, String gameId, int turnIndex, int actionIndex
    ) {
        int[] before = resourceVector(player);
        if ("build_road".equals(kind)) {
            int x = extractInt(responseLine, "\"edge_x\"");
            int y = extractInt(responseLine, "\"edge_y\"");
            int o = extractInt(responseLine, "\"edge_o\"");
            if (x < 0 || y < 0 || o < 0) return false;
            if (!canBuildRoad(player)) return false;
            if (!game.placeRoad(player, new EdgeLocation(x, y, o))) return false;
            int buyCode = game.buyRoad(player);
            if (buyCode != 0) return false;
            int[] after = resourceVector(player);
            assertResourceDelta(after, before, new int[]{-1, 0, 0, 0, -1}, "build_road[ext]");
            logAttempt(config, gameId, turnIndex, actionIndex, player, kind, true, "external", before, after);
            return true;
        }
        if ("build_settlement".equals(kind)) {
            int x = extractInt(responseLine, "\"vertex_x\"");
            int y = extractInt(responseLine, "\"vertex_y\"");
            int o = extractInt(responseLine, "\"vertex_o\"");
            if (x < 0 || y < 0 || o < 0) return false;
            if (!canBuildSettlement(player)) return false;
            if (!game.placeSettlement(player, new VertexLocation(x, y, o))) return false;
            int buyCode = game.buySettlement(player);
            if (buyCode != 0) return false;
            int[] after = resourceVector(player);
            assertResourceDelta(after, before, new int[]{-1, -1, 0, -1, -1}, "build_settlement[ext]");
            logAttempt(config, gameId, turnIndex, actionIndex, player, kind, true, "external", before, after);
            return true;
        }
        if ("build_city".equals(kind)) {
            int x = extractInt(responseLine, "\"vertex_x\"");
            int y = extractInt(responseLine, "\"vertex_y\"");
            int o = extractInt(responseLine, "\"vertex_o\"");
            if (x < 0 || y < 0 || o < 0) return false;
            if (!canBuildCity(player)) return false;
            // Upgrade in-place: Board.placeStructure detects existing settlement.
            if (!game.placeCity(player, new VertexLocation(x, y, o))) return false;
            int buyCode = game.buyCity(player);
            if (buyCode != 0) return false;
            int[] after = resourceVector(player);
            assertResourceDelta(after, before, new int[]{0, 0, -3, -2, 0}, "build_city[ext]");
            logAttempt(config, gameId, turnIndex, actionIndex, player, kind, true, "external", before, after);
            return true;
        }
        if ("buy_dev_card".equals(kind)) {
            int buyCode = game.buyDevCard(player);
            if (buyCode != 0) return false;
            int[] after = resourceVector(player);
            assertResourceDelta(after, before, new int[]{0, -1, -1, -1, 0}, "buy_dev_card[ext]");
            logAttempt(config, gameId, turnIndex, actionIndex, player, kind, true, "external", before, after);
            return true;
        }
        if ("trade_bank".equals(kind)) {
            // For now use the built-in bank trade heuristic (params not
            // parsed).  A future revision can honour give/take/qty.
            boolean success = performBankTrade(game, player);
            int[] after = resourceVector(player);
            if (success) {
                logAttempt(config, gameId, turnIndex, actionIndex, player, kind, true, "external", before, after);
                return true;
            }
            return false;
        }
        return false;
    }

    /** External-policy robber placement: ask Python for the tile. */
    private static void maybeMoveRobberExternal(
        Game game, ArrayList<Player> players, Player current, Random rng,
        ExternalPolicyClient client, Config config, int turnIndex, String gameId
    ) {
        int reqId = client.nextId();
        List<String> legal = Collections.singletonList("move_robber");
        String req = buildDecisionRequest(
            reqId, "robber", players, current, game, gameId, turnIndex, 7, legal
        );
        String responseLine;
        try {
            responseLine = client.exchange(req);
        } catch (IOException e) {
            System.err.println("[BRIDGE] robber request " + reqId + " I/O error: " + e.getMessage());
            client.fallbackCount++;
            maybeMoveRobber(game, rng);
            return;
        }
        String kind = extractStringField(responseLine, "action_kind");
        if (!"move_robber".equals(kind)) {
            emitBridgeError(reqId, "expected_move_robber_got:" + kind);
            client.fallbackCount++;
            maybeMoveRobber(game, rng);
            return;
        }
        int tx = extractInt(responseLine, "\"tile_x\"");
        int ty = extractInt(responseLine, "\"tile_y\"");
        if (tx < 0 || ty < 0) {
            emitBridgeError(reqId, "bad_robber_tile");
            client.fallbackCount++;
            maybeMoveRobber(game, rng);
            return;
        }
        if (!game.getBoard().moveRobber(new Location(tx, ty))) {
            emitBridgeError(reqId, "engine_rejected_tile");
            client.fallbackCount++;
            maybeMoveRobber(game, rng);
        }
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
        int rowIndexInGame,
        int turnIndex,
        int currentPlayerId,
        String currentPlayerName,
        String actionTaken,
        List<String> recentActions,
        Integer roll,
        String phase,
        String snapshotLabel,
        int setupStepIndex
    ) {
        Board board = game.getBoard();
        Location robberLoc = board.getRobberLocation();

        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"game_id\":\"").append(escape(gameId)).append("\",");
        sb.append("\"position_id\":\"").append(escape(gameId)).append("_").append(escape(phase)).append("_").append(rowIndexInGame).append("\",");
        sb.append("\"row_index_in_game\":").append(rowIndexInGame).append(",");
        sb.append("\"turn_metadata\":{");
        sb.append("\"turn_index\":").append(turnIndex).append(",");
        sb.append("\"current_player_id\":\"").append(currentPlayerId).append("\",");
        sb.append("\"current_player_name\":\"").append(escape(currentPlayerName)).append("\",");
        sb.append("\"current_phase\":\"").append(escape(phase)).append("\"");
        sb.append("},");
        sb.append("\"action_taken\":\"").append(escape(actionTaken)).append("\",");
        sb.append("\"snapshot_label\":\"").append(escape(snapshotLabel)).append("\",");
        if (setupStepIndex >= 0) {
            sb.append("\"setup_step_index\":").append(setupStepIndex).append(",");
        }
        sb.append("\"roll\":");
        if (roll == null) {
            sb.append("null,");
        } else {
            sb.append(roll.intValue()).append(",");
        }
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
        sb.append("\"current_phase\":\"").append(escape(phase)).append("\",");
        sb.append("\"current_player_id\":\"").append(currentPlayerId).append("\",");
        sb.append("\"current_player_name\":\"").append(escape(currentPlayerName)).append("\",");
        sb.append("\"last_action\":\"").append(escape(actionTaken)).append("\",");
        sb.append("\"snapshot_label\":\"").append(escape(snapshotLabel)).append("\",");
        if (setupStepIndex >= 0) {
            sb.append("\"setup_step_index\":").append(setupStepIndex).append(",");
        }
        sb.append("\"latest_dice_roll\":");
        if (roll == null) {
            sb.append("null,");
        } else {
            sb.append(roll.intValue()).append(",");
        }

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
