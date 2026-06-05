#include "../ds4.h"
#include "../ds4_distributed.h"

#include <errno.h>
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
    const char *listen_host;
    const char *coordinator_layers;
    const char *worker_layers;
    int listen_port;
    int ctx_size;
    int gen_tokens;
    uint32_t prefill_chunk;
    uint32_t prefill_window;
    uint32_t activation_bits;
    bool debug;
} phase4_cfg;

typedef struct {
    int *tokens;
    int token_count;
    double elapsed_sec;
} token_run;

static double now_sec(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}

static void die_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s --model FILE --listen-host IPV4 --listen-port N "
            "[--prompt-file FILE] [--system TEXT] [--ctx N] [--gen-tokens N] "
            "[--prefill-chunk N] [--prefill-window N] [--activation-bits N] "
            "[--coordinator-layers A:B|A:output] [--worker-layers A:B|A:output] [--no-debug]\n",
            argv0);
    exit(2);
}

static void die(const char *msg) {
    fprintf(stderr, "issue304-phase4-handoff: %s\n", msg);
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
            fprintf(stderr, "issue304-phase4-handoff: route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return -1;
        }
        usleep(100000);
    }
    fprintf(stderr, "issue304-phase4-handoff: timed out waiting for route\n");
    return 0;
}

static bool capture_greedy_trace(ds4_session *session, int steps, token_run *out) {
    memset(out, 0, sizeof(*out));
    out->tokens = malloc((size_t)steps * sizeof(out->tokens[0]));
    if (!out->tokens) return false;

    char err[160];
    double t0 = now_sec();
    for (int i = 0; i < steps; i++) {
        const int token = ds4_session_argmax(session);
        out->tokens[i] = token;
        out->token_count++;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            fprintf(stderr, "issue304-phase4-handoff: distributed eval failed at step %d: %s\n",
                    i, err[0] ? err : "unknown error");
            return false;
        }
    }
    out->elapsed_sec = now_sec() - t0;
    return true;
}

static void free_token_run(token_run *run) {
    free(run->tokens);
    memset(run, 0, sizeof(*run));
}

static void parse_args(phase4_cfg *cfg, int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc) {
            cfg->model_path = argv[++i];
        } else if (!strcmp(argv[i], "--prompt-file") && i + 1 < argc) {
            cfg->prompt_file = argv[++i];
        } else if (!strcmp(argv[i], "--system") && i + 1 < argc) {
            cfg->system_prompt = argv[++i];
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
        } else if (!strcmp(argv[i], "--coordinator-layers") && i + 1 < argc) {
            cfg->coordinator_layers = argv[++i];
        } else if (!strcmp(argv[i], "--worker-layers") && i + 1 < argc) {
            cfg->worker_layers = argv[++i];
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

static bool open_distributed_session(const phase4_cfg *cfg,
                                     ds4_engine *engine,
                                     ds4_session **session_out,
                                     char *route_summary,
                                     size_t route_summary_len,
                                     uint32_t *route_hops,
                                     bool *output_on_coordinator) {
    *session_out = NULL;
    if (ds4_session_create(session_out, engine, cfg->ctx_size) != 0 || !*session_out) {
        fprintf(stderr, "issue304-phase4-handoff: failed to create distributed session\n");
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
        fprintf(stderr, "issue304-phase4-handoff: route summary failed: %s\n",
                err[0] ? err : "unknown error");
        return false;
    }
    return true;
}

static void print_token_array(const int *tokens, int n) {
    putchar('[');
    for (int i = 0; i < n; i++) {
        if (i != 0) printf(",");
        printf("%d", tokens[i]);
    }
    putchar(']');
}

int main(int argc, char **argv) {
    phase4_cfg cfg = {
        .system_prompt = "You are a concise assistant.",
        .ctx_size = 16384,
        .gen_tokens = 16,
        .prefill_chunk = 0,
        .prefill_window = 0,
        .activation_bits = 0,
        .debug = true,
        .coordinator_layers = "0:21",
        .worker_layers = "22:output",
    };
    parse_args(&cfg, argc, argv);

    char *prompt_text = cfg.prompt_file ? read_file(cfg.prompt_file) : strdup("Summarize this test.");
    if (!prompt_text) die("failed to read prompt");

    ds4_distributed_layers coordinator_layers = {0};
    ds4_distributed_layers worker_layers = {0};
    if (!parse_layer_spec(cfg.coordinator_layers, &coordinator_layers) ||
        !parse_layer_spec(cfg.worker_layers, &worker_layers)) {
        die("failed to parse coordinator or worker layer range");
    }

    char err[256] = {0};
    if (!set_process_lock_file("/tmp/ds4-phase4-dist.lock", err, sizeof(err))) {
        die(err);
    }

    ds4_engine *dist_engine = NULL;
    ds4_session *handoff_session = NULL;
    ds4_session *reference_session = NULL;
    ds4_tokens prompt = {0};
    token_run dist_run = {0};
    int *local_tokens = NULL;
    int local_generated = -1;
    int local_first_token = -1;
    int dist_first_token = -1;
    bool token_match = false;
    int mismatch_step = -1;
    double prefill_handoff_sec = 0.0;
    double prefill_ref_sec = 0.0;
    double shard_load_sec = 0.0;
    double local_decode_sec = 0.0;
    uint32_t route_hops = 0;
    bool output_on_coordinator = false;
    char route_summary[1024] = {0};

    ds4_engine_options dist_opt = {
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
    if (ds4_dist_prepare_engine_options(&dist_opt.distributed, &dist_opt, err, sizeof(err)) != 0) {
        die(err[0] ? err : "distributed options failed");
    }
    if (ds4_engine_open(&dist_engine, &dist_opt) != 0 || !dist_engine) {
        die("failed to open distributed coordinator engine");
    }

    if (!open_distributed_session(&cfg,
                                  dist_engine,
                                  &handoff_session,
                                  route_summary,
                                  sizeof(route_summary),
                                  &route_hops,
                                  &output_on_coordinator)) {
        die("failed to open handoff session");
    }
    if (output_on_coordinator) {
        die("phase4 handoff requires worker-owned output head");
    }

    ds4_encode_chat_prompt(dist_engine, cfg.system_prompt, prompt_text, DS4_THINK_NONE, &prompt);
    if (prompt.len <= 0) die("prompt tokenization failed");
    if (prompt.len >= cfg.ctx_size) die("prompt exceeds ctx");

    double t0 = now_sec();
    if (ds4_session_sync(handoff_session, &prompt, err, sizeof(err)) != 0) {
        die(err[0] ? err : "distributed prefill failed");
    }
    prefill_handoff_sec = now_sec() - t0;
    local_first_token = ds4_session_argmax(handoff_session);

    local_tokens = malloc((size_t)cfg.gen_tokens * sizeof(local_tokens[0]));
    if (!local_tokens) die("out of memory allocating local token buffer");
    local_generated = ds4_session_distributed_handoff_argmax(handoff_session,
                                                             cfg.gen_tokens,
                                                             local_tokens,
                                                             cfg.gen_tokens,
                                                             &shard_load_sec,
                                                             &local_decode_sec,
                                                             err,
                                                             sizeof(err));
    if (local_generated < 0) {
        die(err[0] ? err : "distributed local handoff failed");
    }

    ds4_session_free(handoff_session);
    handoff_session = NULL;

    if (!open_distributed_session(&cfg,
                                  dist_engine,
                                  &reference_session,
                                  route_summary,
                                  sizeof(route_summary),
                                  &route_hops,
                                  &output_on_coordinator)) {
        die("failed to open reference session");
    }
    t0 = now_sec();
    if (ds4_session_sync(reference_session, &prompt, err, sizeof(err)) != 0) {
        die(err[0] ? err : "distributed reference prefill failed");
    }
    prefill_ref_sec = now_sec() - t0;
    dist_first_token = ds4_session_argmax(reference_session);
    if (!capture_greedy_trace(reference_session, cfg.gen_tokens, &dist_run)) {
        die("failed to capture distributed reference trace");
    }

    token_match = local_generated == dist_run.token_count;
    if (token_match) {
        for (int i = 0; i < local_generated; i++) {
            if (local_tokens[i] != dist_run.tokens[i]) {
                token_match = false;
                mismatch_step = i;
                break;
            }
        }
    }

    printf("{\n");
    printf("  \"tool\":\"issue304_phase4_handoff\",\n");
    printf("  \"model_path\":\"%s\",\n", cfg.model_path);
    printf("  \"prompt_file\":\"%s\",\n", cfg.prompt_file ? cfg.prompt_file : "");
    printf("  \"ctx\":%d,\n", cfg.ctx_size);
    printf("  \"prompt_tokens\":%d,\n", prompt.len);
    printf("  \"gen_tokens\":%d,\n", cfg.gen_tokens);
    printf("  \"prefill_chunk\":%u,\n", cfg.prefill_chunk);
    printf("  \"prefill_window\":%u,\n", cfg.prefill_window);
    printf("  \"activation_bits\":%u,\n", cfg.activation_bits);
    printf("  \"coordinator_layers\":\"%s\",\n", cfg.coordinator_layers);
    printf("  \"worker_layers\":\"%s\",\n", cfg.worker_layers);
    printf("  \"route_hops\":%u,\n", route_hops);
    printf("  \"route_summary\":\"%s\",\n", route_summary);
    printf("  \"output_owner\":\"%s\",\n", output_on_coordinator ? "coordinator" : "worker");
    printf("  \"prefill_handoff_sec\":%.6f,\n", prefill_handoff_sec);
    printf("  \"prefill_ref_sec\":%.6f,\n", prefill_ref_sec);
    printf("  \"shard_load_sec\":%.6f,\n", shard_load_sec);
    printf("  \"local_decode_sec\":%.6f,\n", local_decode_sec);
    printf("  \"local_decode_tok_per_sec\":%.2f,\n",
           local_decode_sec > 0.0 ? (double)local_generated / local_decode_sec : 0.0);
    printf("  \"distributed_decode_sec\":%.6f,\n", dist_run.elapsed_sec);
    printf("  \"distributed_decode_tok_per_sec\":%.2f,\n",
           dist_run.elapsed_sec > 0.0 ? (double)dist_run.token_count / dist_run.elapsed_sec : 0.0);
    printf("  \"handoff_first_token\":%d,\n", local_first_token);
    printf("  \"reference_first_token\":%d,\n", dist_first_token);
    printf("  \"generated\":{\n");
    printf("    \"count\":%d,\n", local_generated);
    printf("    \"match\":%s,\n", token_match ? "true" : "false");
    printf("    \"mismatch_step\":%d,\n", mismatch_step);
    printf("    \"local_tokens\":");
    print_token_array(local_tokens, local_generated);
    printf(",\n");
    printf("    \"distributed_tokens\":");
    print_token_array(dist_run.tokens, dist_run.token_count);
    printf("\n");
    printf("  }\n");
    printf("}\n");

    free(local_tokens);
    free_token_run(&dist_run);
    ds4_tokens_free(&prompt);
    ds4_session_free(reference_session);
    ds4_session_free(handoff_session);
    ds4_engine_close(dist_engine);
    free(prompt_text);
    return 0;
}
