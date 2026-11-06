"""Run a shell command in a JupyterLab terminal on the CCAD JupyterHub, from the Mac mini. (2026-11-06)

    python tools/hub_run.py status
    python tools/hub_run.py run "COMMAND" [--wait SECONDS] [--start] [--close]
    python tools/hub_run.py close "N [N ...]"          (close terminals by name)

The hub is https://research.labs.ccad.unc.edu.ar (jupyterhub.ccad.unc.edu.ar only redirects there; its API
answers 403 to every token). Token: ~/.config/ccad/hub_token (mode 600, made by the user at /hub/token with the
scopes servers!user access:servers!user read:users!user; never in the repo, never sent anywhere else).
`status` prints the server's state. `run` opens a new terminal in the user's running JupyterLab server (Jupyter
server API: POST /user/NAME/api/terminals, then the terminal websocket), types COMMAND and prints what the terminal
shows for --wait seconds (default 20); the terminal is left open, as a pasted command would be, unless --close. Long jobs must be
detached (`setsid nohup ... &`) as in the hub commands we use. --start starts the server first if it is not
running (a fresh container: nothing of ours is running then). Needs websocket-client (pip install websocket-client).
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

import websocket

HUB = "https://research.labs.ccad.unc.edu.ar"
TOKEN = open(os.path.expanduser("~/.config/ccad/hub_token")).read().strip()
HDR = {"Authorization": f"token {TOKEN}"}


def api(method, url, body=None):
    req = urllib.request.Request(url, method=method, headers={**HDR, "Content-Type": "application/json"},
                                 data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=30) as r:
        txt = r.read().decode()
        return r.status, (json.loads(txt) if txt.strip() else None)


def user():
    return api("GET", f"{HUB}/hub/api/user")[1]


def server_state(u):
    s = (u.get("servers") or {}).get("")
    return s


ap = argparse.ArgumentParser()
ap.add_argument("action", choices=["status", "run", "close"])
ap.add_argument("command", nargs="?")
ap.add_argument("--wait", type=float, default=20.0)
ap.add_argument("--start", action="store_true")
ap.add_argument("--close", action="store_true", help="close the terminal afterwards (safe for setsid-detached jobs)")
a = ap.parse_args()

u = user()
s = server_state(u)
if a.action == "status":
    print(f"user {u['name']}: " + ("no server running" if not s else
          f"server ready={s.get('ready')} pending={s.get('pending')} started {s.get('started')} "
          f"last activity {s.get('last_activity')}"))
    sys.exit(0)

if a.action == "close":          # close terminal(s) by name: hub_run.py close "2 3"
    for n in (a.command or "").split():
        api("DELETE", f"{HUB}{s['url']}api/terminals/{n}")
        print("closed terminal", n)
    sys.exit(0)
if not a.command:
    sys.exit("run needs a COMMAND")
if not s or not s.get("ready"):
    if not a.start:
        sys.exit("the server is not running; pass --start to start it (a fresh container)")
    if not s:
        print("starting the server ...", flush=True)
        api("POST", f"{HUB}/hub/api/users/{u['name']}/server")
    for _ in range(60):
        time.sleep(10)
        s = server_state(user())
        if s and s.get("ready"):
            break
    else:
        sys.exit("the server did not become ready in 10 minutes")
    print("server ready", flush=True)

base = f"{HUB}{s['url']}"            # /user/NAME/
_, term = api("POST", f"{base}api/terminals")
name = term["name"]
ws = websocket.create_connection(f"{base.replace('https://', 'wss://')}terminals/websocket/{name}",
                                 header=[f"Authorization: token {TOKEN}"], timeout=5)
print(f"terminal {name} opened; sending the command", flush=True)
time.sleep(1.0)
ws.send(json.dumps(["stdin", a.command + "\r"]))
ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07")
t_end = time.time() + a.wait
while time.time() < t_end:
    try:
        msg = json.loads(ws.recv())
    except websocket.WebSocketTimeoutException:
        continue
    if msg[0] == "stdout":
        sys.stdout.write(ansi.sub("", msg[1]))
        sys.stdout.flush()
ws.close()
if a.close:
    api("DELETE", f"{base}api/terminals/{name}")
    print(f"\n[terminal {name} closed]")
else:
    print(f"\n[terminal {name} left open]")
