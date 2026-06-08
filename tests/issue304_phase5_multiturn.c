#include "../ds4.h"
#include "../ds4_distributed.h"

#include <errno.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

typedef struct {
    const char *model_path;
    const char *prompt_file;
    const char *system_prompt;
    const char *followup_prompt;
    const char *listen_host;
    const char *coordinator_layers;
    const char *worker_layers;
    const char *lock_file;
    int listen_port;
    int ctx_size;
    int gen_tokens;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    float temperature;
    float top_p;
    float min_p;
    uint64_t seed;
    bool normal_eval;
    bool debug;
} phase5_cfg;

typedef struct {
    int *tokens;
    int token_count;
    double shard_load_sec;
    double decode_sec;
} handoff_run;

typedef struct {
    int top1_ref;
    int top1_cand;
    int top5_overlap;
    int top10_overlap;
    int top20_overlap;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
} parity_metrics;

typedef struct {
    bool enabled;
    ds4_session *session;
    float *turn1_logits;
    float *seed_logits;
} replay_diag;

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void die_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model FILE --listen-host IPV4 --listen-port N "
            "[--prompt-file FILE] [--system TEXT] [--followup TEXT] [--ctx N] "
            "[--gen-tokens N] [--prefill-chunk N] [--prefill-window N] "
            "[--temp F] [--top-p F] [--min-p F] [--seed N] [--normal-eval] "
            "[--activation-bits N] [--coordinator-layers A:B|A:output] "
            "[--worker-layers A:B|A:output] [--lock-file FILE] [--no-debug]\n",
            argv0);
    exit(2);
}

static void die(const char *msg) {
    fprintf(stderr, "issue304-phase5-multiturn: %s\n", msg);
    exit(1);
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *buf = malloc((size_t)len + 1u);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    if (len != 0 && fread(buf, 1, (size_t)len, fp) != (size_t)len) {
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[len] = '\0';
    return buf;
}

static bool parse_layer_spec(const char *spec, ds4_distributed_layers *out) {
    if (!spec || !out) return false;
    const char *colon = strchr(spec, ':');
    if (!colon || colon == spec || colon[1] == '\0') return false;

    errno = 0;
    char *end = NULL;
    unsigned long start = strtoul(spec, &end, 10);
    if (errno != 0 || end != colon || start > UINT32_MAX) return false;

    ds4_distributed_layers layers = {0};
    layers.start = (uint32_t)start;
    layers.set = true;
    if (strcmp(colon + 1, "output") == 0) {
        layers.has_output = true;
        layers.end = 0;
    } else {
        errno = 0;
        unsigned long layer_end = strtoul(colon + 1, &end, 10);
        if (errno != 0 || *end != '\0' || layer_end > UINT32_MAX) return false;
        layers.end = (uint32_t)layer_end;
    }
    *out = layers;
    return true;
}

static bool set_process_lock_file(const char *path, char *err, size_t errlen) {
    if (!path || !path[0]) {
        snprintf(err, errlen, "invalid lock file path");
        return false;
    }
    if (setenv("DS4_LOCK_FILE", path, 1) != 0) {
        snprintf(err, errlen, "failed to set DS4_LOCK_FILE: %s", strerror(errno));
        return false;
    }
    return true;
}

static int wait_route(ds4_session *session, double timeout_sec) {
    double deadline = now_sec() + timeout_sec;
    while (now_sec() < deadline) {
        char err[256] = {0};
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready == 1) return 1;
        if (ready < 0) {
            fprintf(stderr, "issue304-phase5-multiturn: route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return -1;
        }
        usleep(100000);
    }
    fprintf(stderr, "issue304-phase5-multiturn: timed out waiting for route\n");
    return 0;
}

static bool open_distributed_session(const phase5_cfg *cfg,
                                     ds4_engine *engine,
                                     ds4_session **session_out,
                                     char *route_summary,
                                     size_t route_summary_len,
                                     uint32_t *route_hops,
                                     bool *output_on_coordinator) {
    *session_out = NULL;
    if (ds4_session_create(session_out, engine, cfg->ctx_size) != 0 || !*session_out) {
        fprintf(stderr, "issue304-phase5-multiturn: failed to create distributed session\n");
        return false;
    }
    if (wait_route(*session_out, 60.0) != 1) return false;

    char err[256] = {0};
    if (ds4_session_distributed_route_summary(*session_out,
                                              route_summary,
                                              route_summary_len,
                                              route_hops,
                                              output_on_coordinator,
                                              err,
                                              sizeof(err)) != 1) {
        fprintf(stderr, "issue304-phase5-multiturn: route summary failed: %s\n",
                err[0] ? err : "unknown error");
        return false;
    }
    return true;
}

static void handoff_run_free(handoff_run *run) {
    free(run->tokens);
    memset(run, 0, sizeof(*run));
}

static bool run_handoff(ds4_session *session, int max_tokens, handoff_run *out) {
    char err[256] = {0};
    memset(out, 0, sizeof(*out));
    out->tokens = calloc((size_t)max_tokens, sizeof(out->tokens[0]));
    if (!out->tokens) {
        fprintf(stderr, "issue304-phase5-multiturn: out of memory allocating handoff tokens\n");
        return false;
    }

    const int rc = ds4_session_distributed_handoff_argmax(session,
                                                          max_tokens,
                                                          out->tokens,
                                                          max_tokens,
                                                          &out->shard_load_sec,
                                                          &out->decode_sec,
                                                          err,
                                                          sizeof(err));
    if (rc < 0) {
        fprintf(stderr, "issue304-phase5-multiturn: handoff failed: %s\n",
                err[0] ? err : "unknown error");
        return false;
    }
    out->token_count = rc;
    return true;
}

static bool run_normal_eval(ds4_engine *engine,
                            ds4_session *session,
                            const phase5_cfg *cfg,
                            handoff_run *out) {
    char err[256] = {0};
    memset(out, 0, sizeof(*out));
    out->tokens = calloc((size_t)cfg->gen_tokens, sizeof(out->tokens[0]));
    if (!out->tokens) {
        fprintf(stderr, "issue304-phase5-multiturn: out of memory allocating eval tokens\n");
        return false;
    }
    uint64_t rng = cfg->seed;
    for (int i = 0; i < cfg->gen_tokens; i++) {
        int tok = ds4_session_sample(session,
                                     cfg->temperature,
                                     0,
                                     cfg->top_p,
                                     cfg->min_p,
                                     &rng);
        out->tokens[i] = tok;
        out->token_count = i + 1;
        if (tok == ds4_token_eos(engine)) break;
        if (ds4_session_eval(session, tok, err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase5-multiturn: normal eval failed: %s\n",
                    err[0] ? err : "unknown error");
            return false;
        }
    }
    return true;
}

static bool tokens_equal(const int *a, int na, const int *b, int nb) {
    if (na != nb) return false;
    for (int i = 0; i < na; i++) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

static void logits_topk(const float *logits, int n, int *out, int k) {
    for (int i = 0; i < k; i++) out[i] = -1;
    for (int i = 0; i < n; i++) {
        const float v = logits[i];
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || v > logits[out[j]]) {
                for (int t = k - 1; t > j; t--) out[t] = out[t - 1];
                out[j] = i;
                break;
            }
        }
    }
}

static bool topk_contains(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return true;
    }
    return false;
}

static parity_metrics compare_logits(const float *ref, const float *cand, int vocab) {
    int ref_top[64];
    int cand_top[64];
    logits_topk(ref, vocab, ref_top, 64);
    logits_topk(cand, vocab, cand_top, 64);

    int top5_overlap = 0;
    int top10_overlap = 0;
    int top20_overlap = 0;
    for (int i = 0; i < 20; i++) {
        if (ref_top[i] < 0) continue;
        if (i < 5 && topk_contains(cand_top, 5, ref_top[i])) top5_overlap++;
        if (i < 10 && topk_contains(cand_top, 10, ref_top[i])) top10_overlap++;
        if (topk_contains(cand_top, 20, ref_top[i])) top20_overlap++;
    }

    int nonfinite = 0;
    double sumsq = 0.0;
    float max_abs = 0.0f;
    float top20_max_abs = 0.0f;
    for (int i = 0; i < vocab; i++) {
        if (!isfinite(ref[i]) || !isfinite(cand[i])) {
            nonfinite++;
            continue;
        }
        const float delta = cand[i] - ref[i];
        const float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
    }
    for (int i = 0; i < 20; i++) {
        const int id = ref_top[i];
        if (id < 0) continue;
        const float abs_delta = fabsf(cand[id] - ref[id]);
        if (abs_delta > top20_max_abs) top20_max_abs = abs_delta;
    }

    parity_metrics out = {
        .top1_ref = ref_top[0],
        .top1_cand = cand_top[0],
        .top5_overlap = top5_overlap,
        .top10_overlap = top10_overlap,
        .top20_overlap = top20_overlap,
        .nonfinite = nonfinite,
        .rms = (float)sqrt(sumsq / (double)vocab),
        .max_abs = max_abs,
        .top20_max_abs = top20_max_abs,
    };
    return out;
}

static char *token_piece_text(ds4_engine *engine, int token) {
    size_t len = 0;
    char *piece = ds4_token_text(engine, token, &len);
    if (!piece) return strdup("");
    char *out = malloc(len + 1u);
    if (!out) {
        free(piece);
        return NULL;
    }
    memcpy(out, piece, len);
    out[len] = '\0';
    free(piece);
    return out;
}

static void print_json_string(const char *s) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        unsigned char c = *p;
        switch (c) {
            case '\\': fputs("\\\\", stdout); break;
            case '"': fputs("\\\"", stdout); break;
            case '\b': fputs("\\b", stdout); break;
            case '\f': fputs("\\f", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:
                if (c < 0x20u) printf("\\u%04x", c);
                else putchar((char)c);
                break;
        }
    }
    putchar('"');
}

static void print_topk_json(ds4_engine *engine, const float *logits, int vocab, int k) {
    int top[16];
    if (k > (int)(sizeof(top) / sizeof(top[0]))) k = (int)(sizeof(top) / sizeof(top[0]));
    logits_topk(logits, vocab, top, k);
    putchar('[');
    for (int i = 0; i < k; i++) {
        char *piece = token_piece_text(engine, top[i]);
        if (i != 0) putchar(',');
        printf("{\"id\":%d,\"logit\":%.6f,\"text\":", top[i], logits[top[i]]);
        print_json_string(piece ? piece : "");
        printf("}");
        free(piece);
    }
    putchar(']');
}

static char *tokens_render_text(ds4_engine *engine, const int *tokens, int n) {
    size_t cap = 128;
    size_t len = 0;
    char *buf = malloc(cap);
    if (!buf) return NULL;
    buf[0] = '\0';
    for (int i = 0; i < n; i++) {
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, tokens[i], &piece_len);
        if (!piece) continue;
        if (len + piece_len + 1u > cap) {
            size_t need = len + piece_len + 1u;
            size_t next = cap;
            while (next < need) next *= 2u;
            char *grown = realloc(buf, next);
            if (!grown) {
                free(piece);
                free(buf);
                return NULL;
            }
            buf = grown;
            cap = next;
        }
        memcpy(buf + len, piece, piece_len);
        len += piece_len;
        buf[len] = '\0';
        free(piece);
    }
    return buf;
}

static void print_tokens(const int *tokens, int n) {
    putchar('[');
    for (int i = 0; i < n; i++) {
        if (i != 0) putchar(',');
        printf("%d", tokens[i]);
    }
    putchar(']');
}

static int visible_token_count(ds4_engine *engine, const int *tokens, int n) {
    const int eos = ds4_token_eos(engine);
    for (int i = 0; i < n; i++) {
        if (tokens[i] == eos) return i;
    }
    return n;
}

static void build_followup_prompt(ds4_engine *engine,
                                  const ds4_tokens *base_prompt,
                                  const int *assistant_tokens,
                                  int assistant_count,
                                  const char *followup_prompt,
                                  ds4_tokens *turn1_out,
                                  ds4_tokens *out) {
    ds4_tokens_copy(turn1_out, base_prompt);
    for (int i = 0; i < assistant_count; i++) {
        ds4_tokens_push(turn1_out, assistant_tokens[i]);
    }
    ds4_tokens_copy(out, turn1_out);
    ds4_chat_append_message(engine, out, "user", followup_prompt);
    ds4_chat_append_assistant_prefix(engine, out, DS4_THINK_NONE);
}

static void parse_args(phase5_cfg *cfg, int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            cfg->model_path = argv[++i];
        } else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) {
            cfg->prompt_file = argv[++i];
        } else if (!strcmp(argv[i], "--system") && i + 1 < argc) {
            cfg->system_prompt = argv[++i];
        } else if (!strcmp(argv[i], "--followup") && i + 1 < argc) {
            cfg->followup_prompt = argv[++i];
        } else if (!strcmp(argv[i], "--listen-host") && i + 1 < argc) {
            cfg->listen_host = argv[++i];
        } else if (!strcmp(argv[i], "--listen-port") && i + 1 < argc) {
            cfg->listen_port = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            cfg->ctx_size = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--gen-tokens") && i + 1 < argc) {
            cfg->gen_tokens = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--prefill-chunk") && i + 1 < argc) {
            cfg->prefill_chunk = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--prefill-window") && i + 1 < argc) {
            cfg->prefill_window = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--activation-bits") && i + 1 < argc) {
            cfg->activation_bits = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--temp") && i + 1 < argc) {
            cfg->temperature = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--top-p") && i + 1 < argc) {
            cfg->top_p = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--min-p") && i + 1 < argc) {
            cfg->min_p = strtof(argv[++i], NULL);
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            cfg->seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--normal-eval")) {
            cfg->normal_eval = true;
        } else if (!strcmp(argv[i], "--coordinator-layers") && i + 1 < argc) {
            cfg->coordinator_layers = argv[++i];
        } else if (!strcmp(argv[i], "--worker-layers") && i + 1 < argc) {
            cfg->worker_layers = argv[++i];
        } else if (!strcmp(argv[i], "--lock-file") && i + 1 < argc) {
            cfg->lock_file = argv[++i];
        } else if (!strcmp(argv[i], "--no-debug")) {
            cfg->debug = false;
        } else {
            die_usage(argv[0]);
        }
    }

    if (!cfg->model_path || !cfg->listen_host || cfg->listen_port <= 0) {
        die_usage(argv[0]);
    }
}

int main(int argc, char **argv) {
    phase5_cfg cfg = {
        .system_prompt = "You are a concise assistant.",
        .followup_prompt = "Continue with one more short sentence that depends on your previous answer.",
        .ctx_size = 16384,
        .gen_tokens = 8,
        .prefill_chunk = 0,
        .prefill_window = 0,
        .activation_bits = 0,
        .temperature = 0.0f,
        .top_p = 1.0f,
        .min_p = 0.0f,
        .seed = 1u,
        .debug = true,
        .coordinator_layers = "0:21",
        .worker_layers = "22:output",
        .lock_file = "/tmp/ds4-phase5-multiturn-coordinator.lock",
    };
    parse_args(&cfg, argc, argv);

    char *prompt_text = cfg.prompt_file ? read_file(cfg.prompt_file) :
        strdup("Write one short sentence about distributed systems.");
    if (!prompt_text) die("failed to read prompt");

    ds4_distributed_layers coordinator_layers = {0};
    ds4_distributed_layers worker_layers = {0};
    if (!parse_layer_spec(cfg.coordinator_layers, &coordinator_layers) ||
        !parse_layer_spec(cfg.worker_layers, &worker_layers)) {
        die("invalid layer range");
    }

    char err[256] = {0};
    if (!set_process_lock_file(cfg.lock_file, err, sizeof(err))) {
        fprintf(stderr, "issue304-phase5-multiturn: %s\n", err);
        free(prompt_text);
        return 1;
    }

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        .quality = false,
        .distributed = {
            .role = DS4_DISTRIBUTED_COORDINATOR,
            .layers = coordinator_layers,
            .listen_host = cfg.listen_host,
            .listen_port = cfg.listen_port,
            .prefill_chunk = cfg.prefill_chunk,
            .prefill_window = cfg.prefill_window,
            .activation_bits = cfg.activation_bits,
            .debug = cfg.debug,
        },
    };

    if (ds4_dist_prepare_engine_options(&opt.distributed, &opt, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase5-multiturn: distributed options failed: %s\n",
                err[0] ? err : "unknown error");
        free(prompt_text);
        return 1;
    }

    ds4_engine *engine = NULL;
    ds4_session *reused = NULL;
    ds4_session *fresh = NULL;
    replay_diag replay = {
        .enabled = getenv("DS4_PHASE5_DIAG_REPLAY") != NULL,
    };
    ds4_tokens prompt1 = {0};
    ds4_tokens prompt1_turn1 = {0};
    ds4_tokens prompt2 = {0};
    handoff_run turn1 = {0};
    handoff_run turn2_reused = {0};
    handoff_run turn2_fresh = {0};
    float *reused_turn1_logits = NULL;
    float *fresh_turn1_logits = NULL;
    float *reused_seed_logits = NULL;
    float *fresh_seed_logits = NULL;
    int reused_turn1_argmax = -1;
    int reused_argmax = -1;
    int rc = 1;

    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        fprintf(stderr, "issue304-phase5-multiturn: failed to open engine\n");
        goto cleanup;
    }

    char route_summary[256] = {0};
    uint32_t route_hops = 0;
    bool output_on_coordinator = false;
    if (!open_distributed_session(&cfg,
                                  engine,
                                  &reused,
                                  route_summary,
                                  sizeof(route_summary),
                                  &route_hops,
                                  &output_on_coordinator)) {
        goto cleanup;
    }
    if (output_on_coordinator) {
        fprintf(stderr, "issue304-phase5-multiturn: expected worker-owned output head\n");
        goto cleanup;
    }

    ds4_encode_chat_prompt(engine, cfg.system_prompt, prompt_text, DS4_THINK_NONE, &prompt1);
    if (prompt1.len <= 0) {
        fprintf(stderr, "issue304-phase5-multiturn: prompt tokenization failed\n");
        goto cleanup;
    }

    printf("PHASE5_MULTITURN_ROUTE summary=%s hops=%u prompt1_tokens=%d\n",
           route_summary, route_hops, prompt1.len);

    if (ds4_session_sync(reused, &prompt1, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase5-multiturn: first sync failed: %s\n",
                err[0] ? err : "unknown error");
        goto cleanup;
    }

    if (!(cfg.normal_eval
            ? run_normal_eval(engine, reused, &cfg, &turn1)
            : run_handoff(reused, cfg.gen_tokens, &turn1))) goto cleanup;
    const int turn1_visible = visible_token_count(engine, turn1.tokens, turn1.token_count);
    if (turn1_visible <= 0) {
        fprintf(stderr, "issue304-phase5-multiturn: first handoff returned no reusable tokens\n");
        goto cleanup;
    }
    printf("PHASE5_MULTITURN_TURN1 tokens=");
    print_tokens(turn1.tokens, turn1_visible);
    printf(" shard_sec=%.6f decode_sec=%.6f checkpoint_pos=%d\n",
           turn1.shard_load_sec,
           turn1.decode_sec,
           ds4_session_pos(reused));

    build_followup_prompt(engine,
                          &prompt1,
                          turn1.tokens,
                          turn1_visible,
                          cfg.followup_prompt,
                          &prompt1_turn1,
                          &prompt2);

    const int expected_after_turn1 = prompt1.len + turn1_visible;
    if (ds4_session_pos(reused) != expected_after_turn1) {
        fprintf(stderr,
                "issue304-phase5-multiturn: catch-up checkpoint mismatch after turn1: have %d want %d\n",
                ds4_session_pos(reused),
                expected_after_turn1);
        goto cleanup;
    }
    const int vocab = ds4_engine_vocab_size(engine);
    reused_turn1_logits = malloc((size_t)vocab * sizeof(reused_turn1_logits[0]));
    fresh_turn1_logits = malloc((size_t)vocab * sizeof(fresh_turn1_logits[0]));
    reused_seed_logits = malloc((size_t)vocab * sizeof(reused_seed_logits[0]));
    fresh_seed_logits = malloc((size_t)vocab * sizeof(fresh_seed_logits[0]));
    if (!reused_turn1_logits || !fresh_turn1_logits ||
        !reused_seed_logits || !fresh_seed_logits) {
        fprintf(stderr, "issue304-phase5-multiturn: out of memory for logits snapshots\n");
        goto cleanup;
    }
    if (ds4_session_copy_logits(reused, reused_turn1_logits, vocab) != vocab) {
        fprintf(stderr, "issue304-phase5-multiturn: failed to copy reused turn1 logits\n");
        goto cleanup;
    }
    reused_turn1_argmax = ds4_session_argmax(reused);

    if (ds4_session_sync(reused, &prompt2, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase5-multiturn: reused-session followup sync failed: %s\n",
                err[0] ? err : "unknown error");
        goto cleanup;
    }
    if (ds4_session_copy_logits(reused, reused_seed_logits, vocab) != vocab) {
        fprintf(stderr, "issue304-phase5-multiturn: failed to copy reused seed logits\n");
        goto cleanup;
    }
    reused_argmax = ds4_session_argmax(reused);

    if (!(cfg.normal_eval
            ? run_normal_eval(engine, reused, &cfg, &turn2_reused)
            : run_handoff(reused, cfg.gen_tokens, &turn2_reused))) {
        goto cleanup;
    }
    printf("PHASE5_MULTITURN_SYNC reused_pos=%d prompt2_tokens=%d\n",
           ds4_session_pos(reused),
           prompt2.len);
    if (ds4_session_pos(reused) != prompt2.len + turn2_reused.token_count) {
        fprintf(stderr, "issue304-phase5-multiturn: reused session did not advance through turn2\n");
        goto cleanup;
    }

    ds4_session_free(reused);
    reused = NULL;
    ds4_engine_close(engine);
    engine = NULL;

    if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
        fprintf(stderr, "issue304-phase5-multiturn: failed to reopen engine for fresh session\n");
        goto cleanup;
    }
    if (!open_distributed_session(&cfg,
                                  engine,
                                  &fresh,
                                  route_summary,
                                  sizeof(route_summary),
                                  &route_hops,
                                  &output_on_coordinator)) {
        goto cleanup;
    }
    if (ds4_session_sync(fresh, &prompt1_turn1, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase5-multiturn: fresh-session turn1 sync failed: %s\n",
                err[0] ? err : "unknown error");
        goto cleanup;
    }
    if (ds4_session_pos(fresh) != prompt1_turn1.len) {
        fprintf(stderr, "issue304-phase5-multiturn: fresh session did not reach turn1 transcript\n");
        goto cleanup;
    }
    if (ds4_session_copy_logits(fresh, fresh_turn1_logits, vocab) != vocab) {
        fprintf(stderr, "issue304-phase5-multiturn: failed to copy fresh turn1 logits\n");
        goto cleanup;
    }
    const int fresh_turn1_argmax = ds4_session_argmax(fresh);
    parity_metrics turn1_cmp = compare_logits(fresh_turn1_logits, reused_turn1_logits, vocab);
    int fresh_turn1_top5[5];
    int reused_turn1_top5[5];
    logits_topk(fresh_turn1_logits, vocab, fresh_turn1_top5, 5);
    logits_topk(reused_turn1_logits, vocab, reused_turn1_top5, 5);
    printf("PHASE5_MULTITURN_TURN1_ARGMAX reused=%d fresh=%d reused_in_fresh_top5=%s fresh_in_reused_top5=%s\n",
           reused_turn1_argmax,
           fresh_turn1_argmax,
           topk_contains(fresh_turn1_top5, 5, reused_turn1_argmax) ? "true" : "false",
           topk_contains(reused_turn1_top5, 5, fresh_turn1_argmax) ? "true" : "false");
    printf("PHASE5_MULTITURN_TURN1_LOGITS top5=%d top10=%d top20=%d rms=%.8f max_abs=%.8f top20_max_abs=%.8f nonfinite=%d\n",
           turn1_cmp.top5_overlap,
           turn1_cmp.top10_overlap,
           turn1_cmp.top20_overlap,
           turn1_cmp.rms,
           turn1_cmp.max_abs,
           turn1_cmp.top20_max_abs,
           turn1_cmp.nonfinite);
    printf("PHASE5_MULTITURN_TURN1_TOP5 fresh=");
    print_topk_json(engine, fresh_turn1_logits, vocab, 5);
    printf(" reused=");
    print_topk_json(engine, reused_turn1_logits, vocab, 5);
    printf("\n");
    if (ds4_session_sync(fresh, &prompt2, err, sizeof(err)) != 0) {
        fprintf(stderr, "issue304-phase5-multiturn: fresh-session followup sync failed: %s\n",
                err[0] ? err : "unknown error");
        goto cleanup;
    }
    if (ds4_session_pos(fresh) != prompt2.len) {
        fprintf(stderr, "issue304-phase5-multiturn: fresh session did not reach full followup prompt\n");
        goto cleanup;
    }
    if (ds4_session_copy_logits(fresh, fresh_seed_logits, vocab) != vocab) {
        fprintf(stderr, "issue304-phase5-multiturn: failed to copy fresh seed logits\n");
        goto cleanup;
    }
    const int fresh_argmax = ds4_session_argmax(fresh);
    parity_metrics seed_cmp = compare_logits(fresh_seed_logits, reused_seed_logits, vocab);
    int fresh_top5[5];
    int reused_top5[5];
    logits_topk(fresh_seed_logits, vocab, fresh_top5, 5);
    logits_topk(reused_seed_logits, vocab, reused_top5, 5);
    printf("PHASE5_MULTITURN_ARGMAX reused=%d fresh=%d reused_in_fresh_top5=%s fresh_in_reused_top5=%s\n",
           reused_argmax,
           fresh_argmax,
           topk_contains(fresh_top5, 5, reused_argmax) ? "true" : "false",
           topk_contains(reused_top5, 5, fresh_argmax) ? "true" : "false");
    printf("PHASE5_MULTITURN_LOGITS top5=%d top10=%d top20=%d rms=%.8f max_abs=%.8f top20_max_abs=%.8f nonfinite=%d\n",
           seed_cmp.top5_overlap,
           seed_cmp.top10_overlap,
           seed_cmp.top20_overlap,
           seed_cmp.rms,
           seed_cmp.max_abs,
           seed_cmp.top20_max_abs,
           seed_cmp.nonfinite);
    printf("PHASE5_MULTITURN_TOP5 fresh=");
    print_topk_json(engine, fresh_seed_logits, vocab, 5);
    printf(" reused=");
    print_topk_json(engine, reused_seed_logits, vocab, 5);
    printf("\n");
    if (!(cfg.normal_eval
            ? run_normal_eval(engine, fresh, &cfg, &turn2_fresh)
            : run_handoff(fresh, cfg.gen_tokens, &turn2_fresh))) {
        goto cleanup;
    }

    if (replay.enabled) {
        ds4_session_free(fresh);
        fresh = NULL;
        ds4_engine_close(engine);
        engine = NULL;

        if (ds4_engine_open(&engine, &opt) != 0 || !engine) {
            fprintf(stderr, "issue304-phase5-multiturn: failed to reopen engine for replay session\n");
            goto cleanup;
        }
        if (!open_distributed_session(&cfg,
                                      engine,
                                      &replay.session,
                                      route_summary,
                                      sizeof(route_summary),
                                      &route_hops,
                                      &output_on_coordinator)) {
            goto cleanup;
        }
        if (ds4_session_sync(replay.session, &prompt1, err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase5-multiturn: replay-session prompt1 sync failed: %s\n",
                    err[0] ? err : "unknown error");
            goto cleanup;
        }
        for (int i = 0; i < turn1_visible; i++) {
            if (ds4_session_eval(replay.session, turn1.tokens[i], err, sizeof(err)) != 0) {
                fprintf(stderr, "issue304-phase5-multiturn: replay-session turn1 eval failed: %s\n",
                        err[0] ? err : "unknown error");
                goto cleanup;
            }
        }
        replay.turn1_logits = malloc((size_t)vocab * sizeof(replay.turn1_logits[0]));
        replay.seed_logits = malloc((size_t)vocab * sizeof(replay.seed_logits[0]));
        if (!replay.turn1_logits || !replay.seed_logits) {
            fprintf(stderr, "issue304-phase5-multiturn: out of memory for replay diagnostics\n");
            goto cleanup;
        }
        if (ds4_session_copy_logits(replay.session, replay.turn1_logits, vocab) != vocab) {
            fprintf(stderr, "issue304-phase5-multiturn: failed to copy replay turn1 logits\n");
            goto cleanup;
        }
        parity_metrics replay_turn1_cmp =
            compare_logits(replay.turn1_logits, reused_turn1_logits, vocab);
        printf("PHASE5_MULTITURN_REPLAY_TURN1_ARGMAX replay=%d reused=%d\n",
               ds4_session_argmax(replay.session),
               reused_turn1_argmax);
        printf("PHASE5_MULTITURN_REPLAY_TURN1_LOGITS top5=%d top10=%d top20=%d rms=%.8f max_abs=%.8f top20_max_abs=%.8f nonfinite=%d\n",
               replay_turn1_cmp.top5_overlap,
               replay_turn1_cmp.top10_overlap,
               replay_turn1_cmp.top20_overlap,
               replay_turn1_cmp.rms,
               replay_turn1_cmp.max_abs,
               replay_turn1_cmp.top20_max_abs,
               replay_turn1_cmp.nonfinite);
        if (ds4_session_sync(replay.session, &prompt2, err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase5-multiturn: replay-session followup sync failed: %s\n",
                    err[0] ? err : "unknown error");
            goto cleanup;
        }
        if (ds4_session_copy_logits(replay.session, replay.seed_logits, vocab) != vocab) {
            fprintf(stderr, "issue304-phase5-multiturn: failed to copy replay seed logits\n");
            goto cleanup;
        }
        parity_metrics replay_seed_cmp =
            compare_logits(replay.seed_logits, reused_seed_logits, vocab);
        printf("PHASE5_MULTITURN_REPLAY_ARGMAX replay=%d reused=%d\n",
               ds4_session_argmax(replay.session),
               reused_argmax);
        printf("PHASE5_MULTITURN_REPLAY_LOGITS top5=%d top10=%d top20=%d rms=%.8f max_abs=%.8f top20_max_abs=%.8f nonfinite=%d\n",
               replay_seed_cmp.top5_overlap,
               replay_seed_cmp.top10_overlap,
               replay_seed_cmp.top20_overlap,
               replay_seed_cmp.rms,
               replay_seed_cmp.max_abs,
               replay_seed_cmp.top20_max_abs,
               replay_seed_cmp.nonfinite);
    }

    printf("PHASE5_MULTITURN_TURN2 reused=");
    print_tokens(turn2_reused.tokens, turn2_reused.token_count);
    printf(" fresh=");
    print_tokens(turn2_fresh.tokens, turn2_fresh.token_count);
    printf("\n");

    if (!tokens_equal(turn2_reused.tokens,
                      turn2_reused.token_count,
                      turn2_fresh.tokens,
                      turn2_fresh.token_count)) {
        char *reused_text = tokens_render_text(engine, turn2_reused.tokens, turn2_reused.token_count);
        char *fresh_text = tokens_render_text(engine, turn2_fresh.tokens, turn2_fresh.token_count);
        printf("PHASE5_MULTITURN_TEXT reused=");
        print_json_string(reused_text ? reused_text : "");
        printf(" fresh=");
        print_json_string(fresh_text ? fresh_text : "");
        printf("\n");
        free(reused_text);
        free(fresh_text);
        fprintf(stderr, "issue304-phase5-multiturn: turn2 mismatch after catch-up reuse\n");
        goto cleanup;
    }

    printf("PHASE5_MULTITURN_RESULT pass turn1_visible=%d turn2_tokens=%d\n",
           turn1_visible, turn2_reused.token_count);
    rc = 0;

cleanup:
    free(replay.seed_logits);
    free(replay.turn1_logits);
    free(fresh_seed_logits);
    free(reused_seed_logits);
    free(fresh_turn1_logits);
    free(reused_turn1_logits);
    handoff_run_free(&turn2_fresh);
    handoff_run_free(&turn2_reused);
    handoff_run_free(&turn1);
    ds4_tokens_free(&prompt2);
    ds4_tokens_free(&prompt1_turn1);
    ds4_tokens_free(&prompt1);
    ds4_session_free(replay.session);
    ds4_session_free(fresh);
    ds4_session_free(reused);
    ds4_engine_close(engine);
    free(prompt_text);
    return rc;
}
