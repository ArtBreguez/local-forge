"""A stand-in for llama-server: parses the flags forge passes, serves /health and /v1/models.

  FAKE_SERVER_LOG       file to append the received command line to
  FAKE_SERVER_LOAD_S    seconds /health answers 503 ("loading") before 200 (default 0.5)
  FAKE_SERVER_OOM_NGL   -ngl above this value: print a Vulkan OOM message and exit 1
  FAKE_SERVER_NEVER     if "1": never become healthy
"""
import http.server
import os
import sys
import threading
import time

OLD_HELP = "-fa, --flash-attn [on|off|auto]\n--fit [on|off]\n--spec-type TYPE\n--threads-batch N\n"


def arg(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default


def main(args):
    if "--help" in args:
        sys.stdout.write(OLD_HELP)
        return 0
    if os.environ.get("FAKE_SERVER_LOG"):
        with open(os.environ["FAKE_SERVER_LOG"], "a") as f:
            f.write(" ".join(args) + "\n")
    ngl = int(arg(args, "-ngl", "0"))
    limit = os.environ.get("FAKE_SERVER_OOM_NGL")
    if limit is not None and ngl > int(limit):
        time.sleep(0.2)
        sys.stderr.write("ggml_vulkan: Device memory allocation of size 1047364864 failed.\n"
                         "llama_model_load: error loading model: unable to allocate Vulkan0 buffer\n")
        return 1
    t0 = time.time()
    load_s = float(os.environ.get("FAKE_SERVER_LOAD_S", "0.5"))
    never = os.environ.get("FAKE_SERVER_NEVER") == "1"

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/health":
                code = 200 if (time.time() - t0 >= load_s and not never) else 503
            elif self.path == "/v1/models":
                code = 200
            else:
                code = 404
            self.send_response(code)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    srv = http.server.ThreadingHTTPServer((arg(args, "--host", "127.0.0.1"), int(arg(args, "--port", "8080"))), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    while True:
        time.sleep(0.2)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
