#!/usr/bin/env python3
"""Notification helper: stdout + log file + optional email.

Email is sent only when SMTP env vars are set (see below). Otherwise,
notifications go to stdout and the rolling log only.

Required env vars for email:
  GLR_SMTP_HOST       e.g. smtp.gmail.com
  GLR_SMTP_PORT       e.g. 587
  GLR_SMTP_USER       sender address (also used as SMTP login)
  GLR_SMTP_PASSWORD   SMTP password / app password
  GLR_NOTIFY_TO       recipient address(es), comma-separated
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import smtplib
import socket
import sys
from email.message import EmailMessage
from pathlib import Path

LOG_PATH = Path(os.environ.get("GLR_NOTIFY_LOG", str(Path.home() / ".cache/gated-lora-notify.log")))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--level", default="info",
                   choices=["info", "warn", "critical"],
                   help="Severity (controls subject line + email gating)")
    p.add_argument("--subject", default="gated-lora supervisor",
                   help="Notification subject")
    p.add_argument("--message", default=None,
                   help="Body text. If omitted, reads from stdin.")
    p.add_argument("--no-email", action="store_true",
                   help="Force log-only delivery even if SMTP is configured")
    return p.parse_args()


def write_log(level: str, subject: str, body: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with open(LOG_PATH, "a") as f:
        f.write(f"[{ts}] [{level.upper()}] {subject}\n")
        for line in body.splitlines():
            f.write(f"  {line}\n")
        f.write("\n")


def send_email(level: str, subject: str, body: str) -> bool:
    host = os.environ.get("GLR_SMTP_HOST")
    if not host:
        return False
    port = int(os.environ.get("GLR_SMTP_PORT", "587"))
    user = os.environ.get("GLR_SMTP_USER")
    pwd = os.environ.get("GLR_SMTP_PASSWORD")
    to_addr = os.environ.get("GLR_NOTIFY_TO")
    if not (user and pwd and to_addr):
        logging.warning("SMTP partially configured — skipping email")
        return False

    msg = EmailMessage()
    msg["Subject"] = f"[GLR/{level.upper()}] {subject}"
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(
        f"Host: {socket.gethostname()}\n"
        f"Time: {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}Z\n\n"
        f"{body}"
    )

    try:
        with smtplib.SMTP(host, port, timeout=10) as smtp:
            smtp.starttls()
            smtp.login(user, pwd)
            smtp.send_message(msg)
        return True
    except Exception as exc:
        logging.error(f"SMTP send failed: {exc}")
        return False


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    body = args.message if args.message is not None else sys.stdin.read()

    write_log(args.level, args.subject, body)
    print(f"[{args.level.upper()}] {args.subject}\n{body}")

    if args.no_email:
        return 0
    if args.level == "info":
        # Don't email on routine info — only warn/critical
        return 0

    sent = send_email(args.level, args.subject, body)
    if not sent:
        logging.info("Email not sent (SMTP unconfigured or failed). Notification logged only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
