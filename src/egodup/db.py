from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path


SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS assets (
  id INTEGER PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL, size_bytes INTEGER NOT NULL,
  duration_seconds REAL, feature_state TEXT NOT NULL, feature_path TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS submissions (
  id TEXT PRIMARY KEY, asset_id INTEGER NOT NULL REFERENCES assets(id), path TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('reference','query')), created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE(path, role)
);
CREATE TABLE IF NOT EXISTS feature_sets (
  asset_id INTEGER NOT NULL REFERENCES assets(id), profile_sha256 TEXT NOT NULL, path TEXT NOT NULL,
  status TEXT NOT NULL, PRIMARY KEY(asset_id, profile_sha256)
);
CREATE TABLE IF NOT EXISTS index_generations (
  id TEXT PRIMARY KEY, profile_sha256 TEXT NOT NULL, state TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS index_membership (
  generation_id TEXT REFERENCES index_generations(id), asset_id INTEGER REFERENCES assets(id),
  PRIMARY KEY(generation_id, asset_id)
);
CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, manifest_digest TEXT, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS run_items (run_id TEXT REFERENCES runs(id), submission_id TEXT, status TEXT, PRIMARY KEY(run_id, submission_id));
"""


def connect(root: Path) -> sqlite3.Connection:
    root.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(root / "catalog.sqlite")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute("INSERT OR IGNORE INTO metadata(key,value) VALUES('schema_version','1')")
    con.commit()
    return con


def add_asset(
    con: sqlite3.Connection, sha: str, size: int, duration: float | None, path: Path, profile: str
) -> tuple[int, str]:
    con.execute(
        "INSERT OR IGNORE INTO assets(sha256,size_bytes,duration_seconds,feature_state) VALUES(?,?,?,'hashed')",
        (sha, size, duration),
    )
    asset_id = int(con.execute("SELECT id FROM assets WHERE sha256=?", (sha,)).fetchone()[0])
    submission_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"reference:{path}"))
    con.execute(
        "INSERT OR IGNORE INTO submissions(id,asset_id,path,role) VALUES(?,?,?,'reference')",
        (submission_id, asset_id, str(path)),
    )
    con.execute(
        "UPDATE assets SET feature_state='features_ready',feature_path=? WHERE id=?",
        (str(path), asset_id),
    )
    con.execute(
        "INSERT OR REPLACE INTO feature_sets(asset_id,profile_sha256,path,status) VALUES(?,?,?,'complete')",
        (asset_id, profile, str(path)),
    )
    con.commit()
    return asset_id, submission_id


def references(con: sqlite3.Connection, profile: str) -> list[sqlite3.Row]:
    return list(
        con.execute(
            """SELECT a.id,a.sha256,a.duration_seconds,s.id submission_id,s.path,f.path feature_path
      FROM assets a JOIN submissions s ON s.asset_id=a.id JOIN feature_sets f ON f.asset_id=a.id
      WHERE s.role='reference' AND f.profile_sha256=? AND f.status='complete' ORDER BY a.id,s.path""",
            (profile,),
        )
    )
