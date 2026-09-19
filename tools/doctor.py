#!/usr/bin/env python3
"""Diagnose what is stopping you from getting started. Standard library only.

    python tools/doctor.py

Runs anywhere a Python 3.8+ interpreter exists -- including conda's `base`,
before any environment has been created -- because it imports nothing that is
not in the standard library. That is the point: the situations this is for are
exactly the ones where you cannot install anything.

It answers three questions in order:

  1. Can *this* interpreter already run the project?  If yes, you may not need
     to create an environment at all.
  2. Is the network configuration sane -- proxies, conda channels, pip index?
  3. Which package sources are actually reachable, directly and through any
     configured proxy?

and then prints the shortest route from where you are to a working setup.
"""

from __future__ import annotations

import os
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

TIMEOUT = 6.0

# (module, pip/conda name, why it is needed, required?)
REQUIREMENTS = [
    ("torch", "torch", "everything", True),
    ("numpy", "numpy", "everything", True),
    ("scipy", "scipy", "HD95 and the perceptual-hash audit", True),
    ("PIL", "pillow", "image IO", True),
    ("yaml", "pyyaml", "configs and manifests", True),
    ("pytest", "pytest", "the test suite", False),
    ("gdown", "gdown", "tools/prepare_data.py --download only", False),
]

SOURCES = [
    ("conda-forge", "conda.anaconda.org"),
    ("PyPI", "pypi.org"),
    ("PyPI files", "files.pythonhosted.org"),
    ("PyTorch wheels", "download.pytorch.org"),
    ("TUNA conda mirror", "mirrors.tuna.tsinghua.edu.cn"),
    ("TUNA PyPI mirror", "pypi.tuna.tsinghua.edu.cn"),
    ("Anaconda defaults", "repo.anaconda.com"),
]

PROXY_VARS = ["http_proxy", "https_proxy", "all_proxy", "no_proxy",
              "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"]


def rule(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _version(mod) -> str:
    return str(getattr(mod, "__version__", "?"))


def check_interpreter() -> dict:
    rule("1. Can this interpreter run the project?")
    print(f"python {sys.version.split()[0]}  ({sys.executable})")
    env = os.environ.get("CONDA_DEFAULT_ENV")
    if env:
        print(f"active conda environment: {env}")

    found, missing, optional_missing = {}, [], []
    for module, package, purpose, required in REQUIREMENTS:
        try:
            mod = __import__(module)
        except Exception:
            (missing if required else optional_missing).append((package, purpose))
            print(f"  [ MISSING ] {package:<8} {'(required)' if required else '(optional)'} -- {purpose}")
            continue
        found[module] = mod
        print(f"  [ present ] {package:<8} {_version(mod)}")

    notes: list[str] = []
    if "torch" in found and "numpy" in found:
        tv, nv = _version(found["torch"]), _version(found["numpy"])
        try:
            torch_major_minor = tuple(int(x) for x in tv.split("+")[0].split(".")[:2])
            numpy_major = int(nv.split(".")[0])
        except ValueError:
            torch_major_minor, numpy_major = (9, 9), 1
        if numpy_major >= 2 and torch_major_minor < (2, 4):
            notes.append(
                f"torch {tv} predates NumPy 2 C-API support but NumPy is {nv}. "
                "This fails at import. Install 'numpy<2'."
            )
    if "torch" in found:
        torch = found["torch"]
        try:
            if torch.cuda.is_available():
                print(f"  CUDA: yes -- {torch.cuda.get_device_name(0)}, "
                      f"built against CUDA {torch.version.cuda}")
            else:
                print("  CUDA: NO -- tests and the smoke run work; training does not")
                notes.append("torch reports no CUDA device. Check the driver, or that this "
                             "is not a CPU-only wheel.")
        except Exception as exc:
            notes.append(f"torch is installed but querying CUDA failed: {exc}")

    usable = not missing
    print()
    if usable:
        print("  => This interpreter can run the project as it is.")
    else:
        print(f"  => Missing {len(missing)} required package(s): "
              f"{', '.join(p for p, _ in missing)}")
    for n in notes:
        print(f"  !! {n}")
    return {"usable": usable, "missing": [p for p, _ in missing],
            "optional_missing": [p for p, _ in optional_missing], "notes": notes}


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except Exception:
        return ""


def check_config() -> dict:
    rule("2. Proxy and channel configuration")
    env_proxies = {v: os.environ[v] for v in PROXY_VARS if os.environ.get(v)}
    if env_proxies:
        for k, v in env_proxies.items():
            print(f"  {k}={v}")
    else:
        print("  no *_proxy environment variables set")

    condarc_proxies: list[str] = []
    for p in [Path.home() / ".condarc", Path("/etc/conda/.condarc"),
              Path(sys.prefix) / ".condarc"]:
        text = _read(p)
        if not text:
            continue
        print(f"  {p} exists")
        if "proxy_servers" in text:
            block = text[text.index("proxy_servers"):].splitlines()[:5]
            condarc_proxies.append(str(p))
            print("    contains proxy_servers:")
            for line in block:
                print(f"      {line}")
        if "channels" in text:
            after = text[text.index("channels"):].splitlines()[1:6]
            chans = [l.strip("- ").strip() for l in after if l.strip().startswith("-")]
            if chans:
                print(f"    channels: {chans}")

    netrc = Path.home() / ".netrc"
    if netrc.exists():
        print(f"  {netrc} exists (conda and curl both read it)")

    pip_conf = [Path.home() / ".pip" / "pip.conf", Path.home() / ".config" / "pip" / "pip.conf"]
    for p in pip_conf:
        text = _read(p)
        if text:
            print(f"  {p} exists")
            for line in text.splitlines():
                if any(k in line for k in ("proxy", "index-url", "trusted-host")):
                    print(f"      {line.strip()}")

    return {"env_proxies": env_proxies, "condarc_proxies": condarc_proxies}


def tcp_ok(host: str, port: int = 443) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=TIMEOUT):
            return True, "reachable"
    except socket.gaierror as exc:
        return False, f"DNS failure ({exc.strerror or exc})"
    except socket.timeout:
        return False, "timed out"
    except OSError as exc:
        return False, (exc.strerror or str(exc))


def https_ok(host: str, use_proxy: bool) -> tuple[bool, str]:
    handler = urllib.request.ProxyHandler({} if not use_proxy else None)
    opener = urllib.request.build_opener(handler)
    req = urllib.request.Request(f"https://{host}/", method="HEAD")
    try:
        with opener.open(req, timeout=TIMEOUT) as r:
            return True, f"HTTP {r.status}"
    except urllib.error.HTTPError as exc:
        return True, f"HTTP {exc.code}"      # a response is a response
    except Exception as exc:
        return False, type(exc).__name__ + (f": {exc}" if str(exc) else "")


def check_network(cfg: dict) -> dict:
    proxied = bool(cfg["env_proxies"] or cfg["condarc_proxies"])
    rule("3. Which package sources are reachable?")

    if proxied:
        for var in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY"):
            url = cfg["env_proxies"].get(var)
            if not url:
                continue
            hostport = url.split("://")[-1].split("@")[-1].rstrip("/")
            host = hostport.split(":")[0]
            port = int(hostport.split(":")[1]) if ":" in hostport else 8080
            ok, why = tcp_ok(host, port)
            mark = "OK" if ok else "DEAD"
            print(f"  proxy {url}  ->  [{mark}] {why}")
            if not ok:
                print("    A configured proxy that refuses connections is the whole problem.")
            break

    print(f"  {'source':<20} {'direct TCP':<24} HTTPS")
    results = {}
    for label, host in SOURCES:
        ok, why = tcp_ok(host)
        hok, hwhy = https_ok(host, use_proxy=False)
        results[label] = ok
        print(f"  {label:<20} {('OK  ' + why if ok else 'FAIL ' + why):<24} "
              f"{'OK ' + hwhy if hok else 'FAIL ' + hwhy}")
    return results


def recommend(iface: dict, cfg: dict, net: dict) -> int:
    rule("What to do next")
    dead_proxy = bool(cfg["env_proxies"] or cfg["condarc_proxies"])
    direct_ok = net.get("conda-forge") or net.get("PyPI")
    tuna_ok = net.get("TUNA conda mirror") or net.get("TUNA PyPI mirror")

    steps: list[str] = []

    if iface["usable"]:
        steps.append(
            "This interpreter ALREADY has everything required. You do not need to create\n"
            "  an environment at all -- run the project right here:\n"
            "      pytest -q\n"
            "      python tools/make_smoke_data.py --out ./_smoke_data\n"
            "      python tools/train.py --config configs/smoke.yaml"
        )
    elif iface["missing"] and set(iface["missing"]) <= {"scipy", "pyyaml", "pillow"}:
        steps.append(
            "This interpreter is nearly there -- it only lacks "
            f"{', '.join(iface['missing'])}.\n"
            "  Those are small pure-ish packages; installing them is far less work than\n"
            "  building a whole environment:\n"
            f"      conda install -c conda-forge {' '.join(iface['missing'])}\n"
            f"      # or: pip install {' '.join(iface['missing'])}"
        )

    if dead_proxy:
        where = []
        if cfg["env_proxies"]:
            where.append("environment variables")
        if cfg["condarc_proxies"]:
            where.append(f"condarc ({', '.join(cfg['condarc_proxies'])})")
        steps.append(
            f"A proxy is configured, in: {' and '.join(where)}.\n"
            "\n"
            "  FIRST, prove which layer is at fault without changing any configuration.\n"
            "  This runs one command with the proxy variables emptied for that command only:\n"
            "\n"
            "      https_proxy= http_proxy= all_proxy= HTTPS_PROXY= HTTP_PROXY= ALL_PROXY= \\\n"
            "          conda env create -f environment.yml\n"
            "\n"
            "  * If that SUCCEEDS, the environment variables were the problem. Remove them\n"
            "    from whatever sets them (~/.bashrc, /etc/environment) so it stays fixed:\n"
            "        unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY\n"
            "  * If it still FAILS with ProxyError, the proxy is coming from conda's own\n"
            "    configuration, which environment variables cannot override:\n"
            "        conda config --show proxy_servers\n"
            "        conda config --remove-key proxy_servers\n"
            "\n"
            "  If a proxy genuinely IS required on this network, set it correctly for conda\n"
            "  and pip separately -- they do not share configuration:\n"
            "        conda config --set proxy_servers.http  http://user:pass@host:port\n"
            "        conda config --set proxy_servers.https http://user:pass@host:port\n"
            "        pip config set global.proxy http://user:pass@host:port"
        )
        if direct_ok:
            steps.append(
                "Note that the package sources above are reachable DIRECTLY from this machine.\n"
                "  So the proxy is not protecting you from anything here -- removing it is very\n"
                "  likely the whole fix."
            )
        else:
            steps.append(
                "No package source is reachable directly either, so a mirror will not help:\n"
                "  you would need the proxy to reach the mirror too. Fix the proxy first."
            )
    elif direct_ok:
        steps.append("Package sources are reachable. Retry: conda env create -f environment.yml")
    elif tuna_ok:
        steps.append(
            "Upstream is unreachable but the Tsinghua mirrors are. Use the mirrored file:\n"
            "      conda env create -f environment-cn.yml"
        )
    else:
        steps.append(
            "Nothing is reachable, with or without a proxy. This is a network problem\n"
            "  outside this project. Note that the dataset half of the workflow needs NO\n"
            "  network at all -- prepare_data --link/--check, freeze_manifest,\n"
            "  verify_manifest and hash_collisions are entirely local -- so you can do all\n"
            "  of step 3 and 4 of the README while you sort it out, provided some\n"
            "  interpreter on this machine already has the packages (see section 1)."
        )

    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")
        print()
    return 0 if (iface["usable"] or direct_ok or tuna_ok) else 1


def main() -> int:
    print("polyptail doctor -- diagnosing your setup")
    iface = check_interpreter()
    cfg = check_config()
    net = check_network(cfg)
    return recommend(iface, cfg, net)


if __name__ == "__main__":
    raise SystemExit(main())
