"""Durable storage for the standalone agent core.

`client_id` identifies the caller/service principal. `project_id` groups that
client's sessions, assets and knowledge. This module has no browser-user,
password, payment, membership or Cookie tables.
"""

import json
import os
import random
import time
import uuid

import psycopg2


DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
DB_NAME = os.getenv("POSTGRES_DB", "lumanova_agent")
DB_USER = os.getenv("POSTGRES_USER", "lumanova")
DB_PASS = os.getenv("POSTGRES_PASSWORD", "")
DEFAULT_PROJECT_ID = "default"


def get_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )


def _now_ts():
    return int(time.time())


def _vector_literal(vector):
    if not isinstance(vector, (list, tuple)) or not vector:
        raise ValueError("embedding vector is empty")
    return "[" + ",".join(str(float(item)) for item in vector) + "]"


def generate_id():
    return int(f"{int(time.time() * 1000)}{random.randint(100, 999)}")


def ensure_core_client(client_id, label=None):
    """Create the caller and its default project once, without user accounts."""
    client_id = int(client_id)
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO core_clients (client_id, label, created_at, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (client_id) DO UPDATE SET updated_at=EXCLUDED.updated_at
            """,
            (client_id, (label or f"client-{client_id}")[:128], now_ts, now_ts),
        )
        cur.execute(
            """
            INSERT INTO core_projects (project_id, client_id, name, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (client_id, project_id) DO NOTHING
            """,
            (DEFAULT_PROJECT_ID, client_id, DEFAULT_PROJECT_ID, now_ts, now_ts),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()
    return client_id


def ensure_core_project(client_id, project_id=DEFAULT_PROJECT_ID, name=None):
    client_id = ensure_core_client(client_id)
    project_id = str(project_id or DEFAULT_PROJECT_ID).strip()[:128] or DEFAULT_PROJECT_ID
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO core_projects (project_id, client_id, name, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (client_id, project_id)
            DO UPDATE SET updated_at=EXCLUDED.updated_at
            """,
            (project_id, client_id, (name or project_id)[:256], now_ts, now_ts),
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()
    return project_id


def ensure_usage_schema():
    """Create only Agent Core state; this name stays for runtime compatibility."""
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS core_clients (
                client_id BIGINT PRIMARY KEY,
                label VARCHAR(128) NOT NULL,
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS core_projects (
                project_id VARCHAR(128) NOT NULL,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                created_at BIGINT NOT NULL,
                updated_at BIGINT NOT NULL,
                PRIMARY KEY (client_id, project_id)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS core_projects_client_idx ON core_projects(client_id, updated_at DESC)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id VARCHAR(64) PRIMARY KEY,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                project_id VARCHAR(128) NOT NULL DEFAULT 'default',
                title TEXT NOT NULL DEFAULT 'New chat',
                chat_mode VARCHAR(24) NOT NULL DEFAULT 'normal',
                coding_workspace_id VARCHAR(64),
                rag_mode VARCHAR(16) NOT NULL DEFAULT 'none',
                rag_collection_id VARCHAR(64),
                created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, deleted_at BIGINT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS chat_sessions_client_project_updated_idx ON chat_sessions(client_id, project_id, updated_at DESC)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id VARCHAR(64) PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                role VARCHAR(24) NOT NULL, content TEXT NOT NULL, model TEXT,
                message_type VARCHAR(24) NOT NULL DEFAULT 'message',
                sequence_no BIGSERIAL NOT NULL, created_at BIGINT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS chat_messages_session_sequence_idx ON chat_messages(session_id, sequence_no ASC)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_mcp_states (
                session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                server_name VARCHAR(128) NOT NULL, state_key VARCHAR(128) NOT NULL DEFAULT 'default',
                state_json TEXT NOT NULL, updated_at BIGINT NOT NULL,
                PRIMARY KEY (session_id, server_name, state_key)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_assets (
                asset_id VARCHAR(64) PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                asset_type VARCHAR(32) NOT NULL DEFAULT 'image', local_path TEXT, public_url TEXT,
                source_asset_id VARCHAR(64), asset_label TEXT, operation_prompt TEXT,
                is_active BOOLEAN NOT NULL DEFAULT FALSE, created_at BIGINT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS chat_assets_session_created_idx ON chat_assets(session_id, created_at DESC)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_projects (
                session_id VARCHAR(64) PRIMARY KEY REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                stage VARCHAR(32) NOT NULL DEFAULT 'concept_planning', status VARCHAR(64) NOT NULL DEFAULT 'collecting',
                active_subject_id VARCHAR(64), project_brief_json TEXT NOT NULL DEFAULT '{}',
                created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_subjects (
                subject_id VARCHAR(64) PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, display_name TEXT NOT NULL, subject_type VARCHAR(32) NOT NULL DEFAULT 'other',
                story_role TEXT, spec_json TEXT NOT NULL DEFAULT '{}', status VARCHAR(24) NOT NULL DEFAULT 'collecting',
                current_asset_id VARCHAR(64), approved_asset_id VARCHAR(64), created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL,
                UNIQUE (session_id, ordinal)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_subject_assets (
                subject_id VARCHAR(64) NOT NULL REFERENCES video_subjects(subject_id) ON DELETE CASCADE,
                asset_id VARCHAR(64) NOT NULL REFERENCES chat_assets(asset_id) ON DELETE CASCADE,
                asset_role VARCHAR(32) NOT NULL, version INTEGER NOT NULL,
                pipeline_eligible BOOLEAN NOT NULL DEFAULT FALSE, inspection_json TEXT NOT NULL DEFAULT '{}',
                created_at BIGINT NOT NULL, PRIMARY KEY (subject_id, asset_id), UNIQUE (subject_id, asset_role, version)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_keyframes (
                keyframe_id VARCHAR(64) PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL, source_beat_index INTEGER NOT NULL, source_beat_fingerprint VARCHAR(64) NOT NULL,
                time_range TEXT, purpose TEXT, script_content TEXT NOT NULL, script_visual TEXT,
                subject_ids_json TEXT NOT NULL DEFAULT '[]', plan_json TEXT NOT NULL DEFAULT '{}',
                status VARCHAR(24) NOT NULL DEFAULT 'planned', current_asset_id VARCHAR(64), approved_asset_id VARCHAR(64),
                created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL,
                UNIQUE (session_id, ordinal), UNIQUE (session_id, source_beat_index)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_keyframe_assets (
                keyframe_id VARCHAR(64) NOT NULL REFERENCES video_keyframes(keyframe_id) ON DELETE CASCADE,
                asset_id VARCHAR(64) NOT NULL REFERENCES chat_assets(asset_id) ON DELETE CASCADE,
                version INTEGER NOT NULL, pipeline_eligible BOOLEAN NOT NULL DEFAULT FALSE,
                inspection_json TEXT NOT NULL DEFAULT '{}', source_subject_ids_json TEXT NOT NULL DEFAULT '[]',
                created_at BIGINT NOT NULL, PRIMARY KEY (keyframe_id, asset_id), UNIQUE (keyframe_id, version)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_jobs (
                job_id VARCHAR(64) PRIMARY KEY,
                session_id VARCHAR(64) NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                status VARCHAR(24) NOT NULL DEFAULT 'queued', model TEXT, error TEXT, usage_json TEXT,
                created_at BIGINT NOT NULL, started_at BIGINT, completed_at BIGINT, updated_at BIGINT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS chat_jobs_client_status_idx ON chat_jobs(client_id, status, updated_at DESC)")
        conn.commit()
    finally:
        cur.close()
        conn.close()


def ensure_rag_schema():
    ensure_usage_schema()
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_collections (
                collection_id VARCHAR(64) PRIMARY KEY,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                project_id VARCHAR(128) NOT NULL DEFAULT 'default', name TEXT NOT NULL, description TEXT,
                status VARCHAR(24) NOT NULL DEFAULT 'ready', file_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, deleted_at BIGINT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_documents (
                document_id VARCHAR(64) PRIMARY KEY,
                collection_id VARCHAR(64) NOT NULL REFERENCES rag_collections(collection_id) ON DELETE CASCADE,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                filename TEXT NOT NULL, stored_path TEXT NOT NULL, content_type TEXT, file_ext VARCHAR(16),
                status VARCHAR(24) NOT NULL DEFAULT 'queued', error TEXT, chunk_count INTEGER NOT NULL DEFAULT 0,
                created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_jobs (
                job_id VARCHAR(64) PRIMARY KEY,
                collection_id VARCHAR(64) NOT NULL REFERENCES rag_collections(collection_id) ON DELETE CASCADE,
                document_id VARCHAR(64) NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                status VARCHAR(24) NOT NULL DEFAULT 'queued', stage VARCHAR(32) NOT NULL DEFAULT 'queued', error TEXT,
                progress_current INTEGER NOT NULL DEFAULT 0, progress_total INTEGER NOT NULL DEFAULT 0,
                created_at BIGINT NOT NULL, started_at BIGINT, completed_at BIGINT, updated_at BIGINT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS rag_chunks (
                chunk_id VARCHAR(64) PRIMARY KEY,
                document_id VARCHAR(64) NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
                collection_id VARCHAR(64) NOT NULL REFERENCES rag_collections(collection_id) ON DELETE CASCADE,
                client_id BIGINT NOT NULL REFERENCES core_clients(client_id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding vector NOT NULL,
                metadata_json TEXT, created_at BIGINT NOT NULL, UNIQUE (document_id, chunk_index)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS rag_chunks_collection_idx ON rag_chunks(collection_id)")
        conn.commit()
        return True
    except Exception as exc:
        conn.rollback()
        print(f"rag schema unavailable: {type(exc).__name__}: {exc}", flush=True)
        return False
    finally:
        cur.close()
        conn.close()


def rag_schema_ready():
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("SELECT to_regclass('rag_chunks') IS NOT NULL")
        return bool(cur.fetchone()[0])
    finally:
        cur.close()
        conn.close()


def _unmetered_usage(*_args, **_kwargs):
    return {"allowed": True, "limit": None, "used": 0, "reset_at": None}


def client_usage_available(*_args, **_kwargs):
    return True


record_client_token_usage = _unmetered_usage
record_client_chat_usage = _unmetered_usage
record_client_image_usage = _unmetered_usage
def chat_session_to_dict(row):
    if not row:
        return None
    if len(row) == 8:
        session_id, client_id, title, chat_mode, rag_mode, rag_collection_id, created_at, updated_at = row
        coding_workspace_id = None
    else:
        session_id, client_id, title, chat_mode, coding_workspace_id, rag_mode, rag_collection_id, created_at, updated_at = row
    chat_mode = normalize_chat_mode(chat_mode)
    rag_mode = normalize_rag_mode(rag_mode)
    if rag_mode != "collection":
        rag_collection_id = None
    return {
        "session_id": session_id,
        "client_id": client_id,
        "title": title,
        "chat_mode": chat_mode,
        "coding_workspace_id": coding_workspace_id,
        "rag": {
            "mode": rag_mode,
            "collection_id": rag_collection_id,
        },
        "created_at": created_at,
        "updated_at": updated_at,
    }


def normalize_chat_mode(mode):
    mode = (mode or "normal").strip().lower()
    return mode if mode in {"normal", "coding", "video_creation"} else "normal"


def normalize_rag_mode(mode):
    mode = (mode or "none").strip().lower()
    return mode if mode in {"none", "all", "collection"} else "none"


def normalize_rag_scope(mode=None, collection_id=None):
    mode = normalize_rag_mode(mode)
    collection_id = (collection_id or "").strip() or None
    if mode == "collection" and not collection_id:
        mode = "none"
    if mode != "collection":
        collection_id = None
    return mode, collection_id


def chat_message_to_dict(row):
    if not row:
        return None
    message_id, session_id, client_id, role, content, model, message_type, sequence_no, created_at = row
    return {
        "message_id": message_id,
        "session_id": session_id,
        "client_id": client_id,
        "role": role,
        "content": content,
        "model": model,
        "message_type": message_type,
        "sequence_no": sequence_no,
        "created_at": created_at,
    }


def get_chat_mcp_state(client_id, session_id, server_name, state_key="default"):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT state_json
        FROM chat_mcp_states
        WHERE client_id=%s AND session_id=%s AND server_name=%s AND state_key=%s
        """,
        (client_id, session_id, server_name, state_key),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def set_chat_mcp_state(client_id, session_id, server_name, state, state_key="default"):
    now_ts = _now_ts()
    payload = json.dumps(state, ensure_ascii=False, default=str)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_mcp_states (
            session_id, client_id, server_name, state_key, state_json, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_id, server_name, state_key)
        DO UPDATE SET
            client_id=EXCLUDED.client_id,
            state_json=EXCLUDED.state_json,
            updated_at=EXCLUDED.updated_at
        """,
        (session_id, client_id, server_name, state_key, payload, now_ts),
    )
    conn.commit()
    cur.close()
    conn.close()
    return state


def delete_chat_mcp_state(client_id, session_id, server_name, state_key="default"):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM chat_mcp_states
        WHERE client_id=%s AND session_id=%s AND server_name=%s AND state_key=%s
        """,
        (client_id, session_id, server_name, state_key),
    )
    deleted = cur.rowcount > 0
    conn.commit()
    cur.close()
    conn.close()
    return deleted


def chat_job_to_dict(row):
    if not row:
        return None
    job_id, session_id, client_id, status, model, error, usage_json, created_at, started_at, completed_at, updated_at = row
    usage = None
    if usage_json:
        try:
            usage = json.loads(usage_json)
        except Exception:
            usage = None
    return {
        "job_id": job_id,
        "session_id": session_id,
        "client_id": client_id,
        "status": status,
        "model": model,
        "error": error,
        "usage": usage,
        "created_at": created_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "updated_at": updated_at,
    }


def create_chat_session(client_id, title=None, rag_mode="none", rag_collection_id=None, chat_mode="normal", coding_workspace_id=None, project_id=DEFAULT_PROJECT_ID):
    client_id = ensure_core_client(client_id)
    project_id = ensure_core_project(client_id, project_id)
    now_ts = int(time.time())
    session_id = f"sess_{uuid.uuid4().hex}"
    title = (title or "新对话").strip()[:80] or "新对话"
    chat_mode = normalize_chat_mode(chat_mode)
    rag_mode, rag_collection_id = normalize_rag_scope(rag_mode, rag_collection_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_sessions (session_id, client_id, project_id, title, chat_mode, coding_workspace_id, rag_mode, rag_collection_id, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING session_id, client_id, title, chat_mode, coding_workspace_id, rag_mode, rag_collection_id, created_at, updated_at
        """,
        (session_id, client_id, project_id, title, chat_mode, coding_workspace_id, rag_mode, rag_collection_id, now_ts, now_ts),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return chat_session_to_dict(row)


def list_chat_sessions(client_id, limit=100, project_id=None):
    project_filter = " AND s.project_id=%s" if project_id else ""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT s.session_id, s.client_id, s.title, s.chat_mode, s.coding_workspace_id, s.rag_mode, s.rag_collection_id, s.created_at, s.updated_at,
               j.job_id, j.status, j.error, j.updated_at,
               vp.stage, vp.status, vp.project_brief_json
        FROM chat_sessions s
        LEFT JOIN LATERAL (
            SELECT job_id, status, error, updated_at
            FROM chat_jobs
            WHERE session_id=s.session_id AND client_id=s.client_id
            ORDER BY updated_at DESC
            LIMIT 1
        ) j ON TRUE
        LEFT JOIN video_projects vp ON vp.session_id=s.session_id AND vp.client_id=s.client_id
        WHERE s.client_id=%s AND s.deleted_at IS NULL{project_filter}
        ORDER BY s.updated_at DESC
        LIMIT %s
        """,
        (client_id, limit) if not project_id else (client_id, project_id, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    sessions = []
    for row in rows:
        session = chat_session_to_dict(row[:9])
        job_id, job_status, job_error, job_updated_at, video_stage, video_status, video_brief_json = row[9:]
        session["job"] = {
            "job_id": job_id,
            "status": job_status,
            "error": job_error,
            "updated_at": job_updated_at,
        } if job_id else None
        video_brief = _json_object(video_brief_json)
        video_task = video_brief.get("video_task") if isinstance(video_brief, dict) else {}
        if video_stage == "video_generating" and isinstance(video_task, dict):
            session["video_generation"] = {
                "status": str(video_task.get("status") or video_status or "running"),
                "gateway_job_id": video_task.get("gateway_job_id"),
                "updated_at": video_task.get("updated_at") or session.get("updated_at"),
            }
        else:
            session["video_generation"] = None
        sessions.append(session)
    return sessions


def get_chat_session(client_id, session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT session_id, client_id, title, chat_mode, coding_workspace_id, rag_mode, rag_collection_id, created_at, updated_at
        FROM chat_sessions
        WHERE client_id=%s AND session_id=%s AND deleted_at IS NULL
        """,
        (client_id, session_id),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return chat_session_to_dict(row)


def update_chat_session(client_id, session_id, title=None, rag_mode=None, rag_collection_id=None, chat_mode=None, coding_workspace_id=None):
    title = (title or "").strip()[:80]
    update_rag = rag_mode is not None or rag_collection_id is not None
    update_chat_mode = chat_mode is not None
    update_coding_workspace = coding_workspace_id is not None
    normalized_chat_mode = normalize_chat_mode(chat_mode)
    normalized_rag_mode, normalized_rag_collection_id = normalize_rag_scope(rag_mode, rag_collection_id)
    now_ts = int(time.time())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT title, chat_mode, coding_workspace_id, rag_mode, rag_collection_id
        FROM chat_sessions
        WHERE client_id=%s AND session_id=%s AND deleted_at IS NULL
        """,
        (client_id, session_id),
    )
    existing = cur.fetchone()
    if not existing:
        conn.rollback()
        cur.close()
        conn.close()
        return None
    next_title = title or existing[0]
    next_chat_mode = normalized_chat_mode if update_chat_mode else normalize_chat_mode(existing[1])
    next_coding_workspace_id = coding_workspace_id if update_coding_workspace else existing[2]
    next_rag_mode = normalized_rag_mode if update_rag else normalize_rag_mode(existing[3])
    next_rag_collection_id = normalized_rag_collection_id if update_rag else existing[4]
    if update_rag:
        next_chat_mode = "normal"
        next_coding_workspace_id = None
    if next_chat_mode != "coding":
        next_coding_workspace_id = None
    if next_chat_mode != "normal":
        next_rag_mode = "none"
        next_rag_collection_id = None
    next_rag_mode, next_rag_collection_id = normalize_rag_scope(next_rag_mode, next_rag_collection_id)
    cur.execute(
        """
        UPDATE chat_sessions
        SET title=%s, chat_mode=%s, coding_workspace_id=%s, rag_mode=%s, rag_collection_id=%s, updated_at=%s
        WHERE client_id=%s AND session_id=%s AND deleted_at IS NULL
        RETURNING session_id, client_id, title, chat_mode, coding_workspace_id, rag_mode, rag_collection_id, created_at, updated_at
        """,
        (next_title, next_chat_mode, next_coding_workspace_id, next_rag_mode, next_rag_collection_id, now_ts, client_id, session_id),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return chat_session_to_dict(row)


def update_chat_session_title(client_id, session_id, title):
    title = (title or "").strip()[:80]
    if not title:
        return None
    return update_chat_session(client_id, session_id, title=title)


def delete_chat_session(client_id, session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        DELETE FROM chat_sessions
        WHERE client_id=%s AND session_id=%s
        """,
        (client_id, session_id),
    )
    conn.commit()
    changed = cur.rowcount > 0
    cur.close()
    conn.close()
    return changed


def chat_asset_to_dict(row):
    if not row:
        return None
    asset_id, session_id, client_id, asset_type, local_path, public_url, source_asset_id, asset_label, operation_prompt, is_active, created_at = row
    return {
        "asset_id": asset_id,
        "session_id": session_id,
        "client_id": client_id,
        "asset_type": asset_type,
        "local_path": local_path,
        "public_url": public_url,
        "source_asset_id": source_asset_id,
        "asset_label": asset_label,
        "operation_prompt": operation_prompt,
        "is_active": bool(is_active),
        "created_at": created_at,
    }


def list_chat_assets(client_id, session_id, limit=12):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT asset_id, session_id, client_id, asset_type, local_path, public_url, source_asset_id,
               asset_label, operation_prompt, is_active, created_at
        FROM chat_assets
        WHERE client_id=%s AND session_id=%s
        ORDER BY is_active DESC, created_at DESC
        LIMIT %s
        """,
        (client_id, session_id, max(int(limit or 12), 1)),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [chat_asset_to_dict(row) for row in rows]


def ensure_chat_asset(
    client_id,
    session_id,
    *,
    asset_type="image",
    local_path=None,
    public_url=None,
    source_asset_id=None,
    asset_label=None,
    operation_prompt=None,
    make_active=True,
):
    """Create or reactivate one session-owned visual asset without duplicates."""
    local_path = (local_path or "").strip() or None
    public_url = (public_url or "").strip() or None
    if not local_path and not public_url:
        return None

    now_ts = int(time.time())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT asset_id, session_id, client_id, asset_type, local_path, public_url, source_asset_id,
               asset_label, operation_prompt, is_active, created_at
        FROM chat_assets
        WHERE client_id=%s AND session_id=%s
          AND ((%s IS NOT NULL AND local_path=%s) OR (%s IS NOT NULL AND public_url=%s))
        ORDER BY created_at DESC
        LIMIT 1
        FOR UPDATE
        """,
        (client_id, session_id, local_path, local_path, public_url, public_url),
    )
    existing = cur.fetchone()
    if make_active:
        cur.execute(
            "UPDATE chat_assets SET is_active=FALSE WHERE client_id=%s AND session_id=%s AND is_active=TRUE",
            (client_id, session_id),
        )

    if existing:
        asset_id = existing[0]
        cur.execute(
            """
            UPDATE chat_assets
            SET asset_type=%s, local_path=COALESCE(%s, local_path), public_url=COALESCE(%s, public_url),
                source_asset_id=COALESCE(%s, source_asset_id),
                asset_label=COALESCE(%s, asset_label), operation_prompt=COALESCE(%s, operation_prompt),
                is_active=%s
            WHERE asset_id=%s
            RETURNING asset_id, session_id, client_id, asset_type, local_path, public_url, source_asset_id,
                      asset_label, operation_prompt, is_active, created_at
            """,
            (asset_type, local_path, public_url, source_asset_id, asset_label, operation_prompt, bool(make_active), asset_id),
        )
    else:
        asset_id = f"asset_{uuid.uuid4().hex}"
        cur.execute(
            """
            INSERT INTO chat_assets (
                asset_id, session_id, client_id, asset_type, local_path, public_url,
                source_asset_id, asset_label, operation_prompt, is_active, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING asset_id, session_id, client_id, asset_type, local_path, public_url, source_asset_id,
                      asset_label, operation_prompt, is_active, created_at
            """,
            (asset_id, session_id, client_id, asset_type, local_path, public_url, source_asset_id,
             asset_label, operation_prompt, bool(make_active), now_ts),
        )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return chat_asset_to_dict(row)


def _json_object(value):
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_list(value):
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def video_project_to_dict(row):
    if not row:
        return None
    session_id, client_id, stage, status, active_subject_id, project_brief_json, created_at, updated_at = row
    return {
        "session_id": session_id,
        "client_id": client_id,
        "stage": stage,
        "status": status,
        "active_subject_id": active_subject_id,
        "project_brief": _json_object(project_brief_json),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def get_or_create_video_project(client_id, session_id):
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO video_projects (
            session_id, client_id, stage, status, project_brief_json, created_at, updated_at
        )
        VALUES (%s, %s, 'concept_planning', 'collecting', '{}', %s, %s)
        ON CONFLICT (session_id) DO NOTHING
        """,
        (session_id, client_id, now_ts, now_ts),
    )
    cur.execute(
        """
        SELECT session_id, client_id, stage, status, active_subject_id,
               project_brief_json, created_at, updated_at
        FROM video_projects
        WHERE client_id=%s AND session_id=%s
        """,
        (client_id, session_id),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return video_project_to_dict(row)


def get_video_project(client_id, session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT session_id, client_id, stage, status, active_subject_id,
               project_brief_json, created_at, updated_at
        FROM video_projects
        WHERE client_id=%s AND session_id=%s
        """,
        (client_id, session_id),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return video_project_to_dict(row)


def list_stale_video_generation_projects(stale_before, limit=50):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT session_id, client_id, stage, status, active_subject_id,
               project_brief_json, created_at, updated_at
        FROM video_projects
        WHERE stage='video_generating' AND updated_at <= %s
        ORDER BY updated_at ASC
        LIMIT %s
        """,
        (int(stale_before), max(1, min(int(limit), 200))),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [video_project_to_dict(row) for row in rows]


def update_video_project(
    client_id,
    session_id,
    *,
    stage=None,
    status=None,
    active_subject_id=None,
    project_brief=None,
):
    project = get_or_create_video_project(client_id, session_id)
    if not project:
        return None
    next_stage = (stage or project["stage"] or "concept_planning").strip()[:32]
    next_status = (status or project["status"] or "collecting").strip()[:64]
    next_active_subject_id = active_subject_id if active_subject_id is not None else project.get("active_subject_id")
    next_brief = project_brief if isinstance(project_brief, dict) else project.get("project_brief") or {}
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE video_projects
        SET stage=%s, status=%s, active_subject_id=%s, project_brief_json=%s, updated_at=%s
        WHERE client_id=%s AND session_id=%s
        RETURNING session_id, client_id, stage, status, active_subject_id,
                  project_brief_json, created_at, updated_at
        """,
        (
            next_stage,
            next_status,
            next_active_subject_id,
            json.dumps(next_brief, ensure_ascii=False, default=str),
            now_ts,
            client_id,
            session_id,
        ),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return video_project_to_dict(row)


def video_subject_to_dict(row):
    if not row:
        return None
    (
        subject_id,
        session_id,
        client_id,
        ordinal,
        display_name,
        subject_type,
        story_role,
        spec_json,
        status,
        current_asset_id,
        approved_asset_id,
        created_at,
        updated_at,
    ) = row
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "client_id": client_id,
        "ordinal": ordinal,
        "display_name": display_name,
        "subject_type": subject_type,
        "story_role": story_role,
        "spec": _json_object(spec_json),
        "status": status,
        "current_asset_id": current_asset_id,
        "approved_asset_id": approved_asset_id,
        "created_at": created_at,
        "updated_at": updated_at,
    }


_VIDEO_SUBJECT_SELECT = """
    SELECT subject_id, session_id, client_id, ordinal, display_name, subject_type,
           story_role, spec_json, status, current_asset_id, approved_asset_id,
           created_at, updated_at
    FROM video_subjects
"""


def list_video_subjects(client_id, session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        _VIDEO_SUBJECT_SELECT + " WHERE client_id=%s AND session_id=%s AND status <> 'removed' ORDER BY ordinal ASC",
        (client_id, session_id),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [video_subject_to_dict(row) for row in rows]


def replace_video_subject_in_roster(
    client_id,
    session_id,
    *,
    outgoing_subject_id,
    incoming_subject_id,
):
    """Replace the canonical concept subject and retire its obsolete material row.

    Subject references are production state, not chat-only labels.  Keeping the
    concept roster and the material roster in one transaction prevents a
    removed subject from silently blocking later stages.
    """
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (session_id,))
    cur.execute(
        _VIDEO_SUBJECT_SELECT + " WHERE client_id=%s AND session_id=%s AND subject_id=%s FOR UPDATE",
        (client_id, session_id, outgoing_subject_id),
    )
    outgoing_row = cur.fetchone()
    cur.execute(
        _VIDEO_SUBJECT_SELECT + " WHERE client_id=%s AND session_id=%s AND subject_id=%s FOR UPDATE",
        (client_id, session_id, incoming_subject_id),
    )
    incoming_row = cur.fetchone()
    cur.execute(
        "SELECT project_brief_json, active_subject_id FROM video_projects WHERE client_id=%s AND session_id=%s FOR UPDATE",
        (client_id, session_id),
    )
    project_row = cur.fetchone()
    if not outgoing_row or not incoming_row or not project_row:
        conn.rollback()
        cur.close()
        conn.close()
        return None

    outgoing = video_subject_to_dict(outgoing_row)
    incoming = video_subject_to_dict(incoming_row)
    brief = _json_object(project_row[0])
    concept = brief.get("concept") if isinstance(brief.get("concept"), dict) else {}
    roster = concept.get("subjects") if isinstance(concept.get("subjects"), list) else []
    replacement = {
        "display_name": incoming.get("display_name") or "主体",
        "subject_type": incoming.get("subject_type") or "other",
        "role": incoming.get("story_role") or "",
    }
    visual_direction = str((incoming.get("spec") or {}).get("visual_direction") or "").strip()
    if visual_direction:
        replacement["visual_direction"] = visual_direction
    replaced = False
    next_roster = []
    for item in roster:
        if not isinstance(item, dict):
            continue
        name = str(item.get("display_name") or "").strip().lower()
        if name == str(outgoing.get("display_name") or "").strip().lower():
            if not replaced:
                next_roster.append(replacement)
                replaced = True
            continue
        next_roster.append(item)
    if not replaced:
        next_roster.append(replacement)
    concept["subjects"] = next_roster
    brief["concept"] = concept

    cur.execute(
        """
        UPDATE video_subjects
        SET status='removed', current_asset_id=NULL, approved_asset_id=NULL, updated_at=%s
        WHERE client_id=%s AND session_id=%s AND subject_id=%s
        """,
        (now_ts, client_id, session_id, outgoing_subject_id),
    )
    active_subject_id = incoming_subject_id if project_row[1] == outgoing_subject_id else project_row[1]
    cur.execute(
        """
        UPDATE video_projects
        SET active_subject_id=%s, status='subject_ready', project_brief_json=%s, updated_at=%s
        WHERE client_id=%s AND session_id=%s
        RETURNING session_id, client_id, stage, status, active_subject_id,
                  project_brief_json, created_at, updated_at
        """,
        (
            active_subject_id,
            json.dumps(brief, ensure_ascii=False, default=str),
            now_ts,
            client_id,
            session_id,
        ),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return {"project": video_project_to_dict(row), "outgoing": outgoing, "incoming": incoming}


def remove_video_subject_from_roster(client_id, session_id, *, subject_id):
    """Retire a subject and remove it from the canonical concept roster."""
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (session_id,))
    cur.execute(
        _VIDEO_SUBJECT_SELECT + " WHERE client_id=%s AND session_id=%s AND subject_id=%s FOR UPDATE",
        (client_id, session_id, subject_id),
    )
    subject_row = cur.fetchone()
    cur.execute(
        "SELECT project_brief_json, active_subject_id FROM video_projects WHERE client_id=%s AND session_id=%s FOR UPDATE",
        (client_id, session_id),
    )
    project_row = cur.fetchone()
    if not subject_row or not project_row:
        conn.rollback()
        cur.close()
        conn.close()
        return None
    subject = video_subject_to_dict(subject_row)
    brief = _json_object(project_row[0])
    concept = brief.get("concept") if isinstance(brief.get("concept"), dict) else {}
    roster = concept.get("subjects") if isinstance(concept.get("subjects"), list) else []
    removed_name = str(subject.get("display_name") or "").strip().lower()
    concept["subjects"] = [
        item for item in roster
        if not isinstance(item, dict) or str(item.get("display_name") or "").strip().lower() != removed_name
    ]
    brief["concept"] = concept
    cur.execute(
        """
        UPDATE video_subjects
        SET status='removed', current_asset_id=NULL, approved_asset_id=NULL, updated_at=%s
        WHERE client_id=%s AND session_id=%s AND subject_id=%s
        """,
        (now_ts, client_id, session_id, subject_id),
    )
    active_subject_id = None if project_row[1] == subject_id else project_row[1]
    cur.execute(
        """
        UPDATE video_projects
        SET active_subject_id=%s, status='subject_ready', project_brief_json=%s, updated_at=%s
        WHERE client_id=%s AND session_id=%s
        RETURNING session_id, client_id, stage, status, active_subject_id,
                  project_brief_json, created_at, updated_at
        """,
        (
            active_subject_id,
            json.dumps(brief, ensure_ascii=False, default=str),
            now_ts,
            client_id,
            session_id,
        ),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return {"project": video_project_to_dict(row), "removed": subject}


def get_video_subject(client_id, session_id, subject_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        _VIDEO_SUBJECT_SELECT + " WHERE client_id=%s AND session_id=%s AND subject_id=%s",
        (client_id, session_id, subject_id),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return video_subject_to_dict(row)


def upsert_video_subject(
    client_id,
    session_id,
    *,
    subject_id=None,
    display_name=None,
    subject_type="other",
    story_role=None,
    spec=None,
    status=None,
):
    now_ts = _now_ts()
    spec = spec if isinstance(spec, dict) else {}
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (session_id,))
    existing = None
    if subject_id:
        cur.execute(
            _VIDEO_SUBJECT_SELECT + " WHERE client_id=%s AND session_id=%s AND subject_id=%s FOR UPDATE",
            (client_id, session_id, subject_id),
        )
        existing = cur.fetchone()
    if not existing and display_name:
        cur.execute(
            _VIDEO_SUBJECT_SELECT
            + " WHERE client_id=%s AND session_id=%s AND lower(display_name)=lower(%s) ORDER BY ordinal ASC LIMIT 1 FOR UPDATE",
            (client_id, session_id, display_name.strip()),
        )
        existing = cur.fetchone()

    if existing:
        current = video_subject_to_dict(existing)
        merged_spec = {**(current.get("spec") or {}), **spec}
        next_name = (display_name or current["display_name"] or "主体").strip()[:80]
        next_type = (subject_type or current["subject_type"] or "other").strip()[:32]
        next_role = (story_role if story_role is not None else current.get("story_role"))
        next_status = (status or current["status"] or "collecting").strip()[:24]
        cur.execute(
            """
            UPDATE video_subjects
            SET display_name=%s, subject_type=%s, story_role=%s, spec_json=%s,
                status=%s, updated_at=%s
            WHERE subject_id=%s
            RETURNING subject_id, session_id, client_id, ordinal, display_name,
                      subject_type, story_role, spec_json, status, current_asset_id,
                      approved_asset_id, created_at, updated_at
            """,
            (
                next_name,
                next_type,
                next_role,
                json.dumps(merged_spec, ensure_ascii=False, default=str),
                next_status,
                now_ts,
                current["subject_id"],
            ),
        )
    else:
        cur.execute(
            "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM video_subjects WHERE client_id=%s AND session_id=%s",
            (client_id, session_id),
        )
        ordinal = int(cur.fetchone()[0] or 1)
        subject_id = f"subject_{uuid.uuid4().hex}"
        next_name = (display_name or f"主体 {ordinal}").strip()[:80]
        next_type = (subject_type or "other").strip()[:32]
        next_status = (status or "collecting").strip()[:24]
        cur.execute(
            """
            INSERT INTO video_subjects (
                subject_id, session_id, client_id, ordinal, display_name, subject_type,
                story_role, spec_json, status, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING subject_id, session_id, client_id, ordinal, display_name,
                      subject_type, story_role, spec_json, status, current_asset_id,
                      approved_asset_id, created_at, updated_at
            """,
            (
                subject_id,
                session_id,
                client_id,
                ordinal,
                next_name,
                next_type,
                story_role,
                json.dumps(spec, ensure_ascii=False, default=str),
                next_status,
                now_ts,
                now_ts,
            ),
        )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return video_subject_to_dict(row)


def video_subject_asset_to_dict(row):
    if not row:
        return None
    subject_id, asset_id, asset_role, version, pipeline_eligible, inspection_json, created_at = row
    return {
        "subject_id": subject_id,
        "asset_id": asset_id,
        "asset_role": asset_role,
        "version": version,
        "pipeline_eligible": bool(pipeline_eligible),
        "inspection": _json_object(inspection_json),
        "created_at": created_at,
    }


def attach_video_subject_asset(
    client_id,
    session_id,
    subject_id,
    asset_id,
    *,
    asset_role,
    inspection=None,
    pipeline_eligible=False,
):
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (subject_id,))
    cur.execute(
        "SELECT 1 FROM video_subjects WHERE client_id=%s AND session_id=%s AND subject_id=%s FOR UPDATE",
        (client_id, session_id, subject_id),
    )
    if not cur.fetchone():
        conn.rollback()
        cur.close()
        conn.close()
        return None
    cur.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM video_subject_assets WHERE subject_id=%s AND asset_role=%s",
        (subject_id, asset_role),
    )
    version = int(cur.fetchone()[0] or 1)
    cur.execute(
        """
        INSERT INTO video_subject_assets (
            subject_id, asset_id, asset_role, version, pipeline_eligible,
            inspection_json, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING subject_id, asset_id, asset_role, version, pipeline_eligible,
                  inspection_json, created_at
        """,
        (
            subject_id,
            asset_id,
            asset_role,
            version,
            bool(pipeline_eligible),
            json.dumps(inspection or {}, ensure_ascii=False, default=str),
            now_ts,
        ),
    )
    row = cur.fetchone()
    if asset_role == "design_sheet":
        cur.execute(
            """
            UPDATE video_subjects
            SET current_asset_id=%s, status='awaiting_review', updated_at=%s
            WHERE subject_id=%s
            """,
            (asset_id, now_ts, subject_id),
        )
    project_status = "awaiting_review" if asset_role == "design_sheet" else "subject_ready"
    cur.execute(
        """
        UPDATE video_projects
        SET active_subject_id=%s, status=%s, updated_at=%s
        WHERE client_id=%s AND session_id=%s
        """,
        (subject_id, project_status, now_ts, client_id, session_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return video_subject_asset_to_dict(row)


def list_video_subject_assets(client_id, session_id, subject_id=None):
    conn = get_conn()
    cur = conn.cursor()
    query = """
        SELECT links.subject_id, links.asset_id, links.asset_role, links.version,
               links.pipeline_eligible, links.inspection_json, links.created_at,
               assets.public_url, assets.local_path, assets.source_asset_id,
               assets.asset_label, assets.operation_prompt
        FROM video_subject_assets AS links
        JOIN video_subjects AS subjects ON subjects.subject_id=links.subject_id
        JOIN chat_assets AS assets ON assets.asset_id=links.asset_id
        WHERE subjects.client_id=%s AND subjects.session_id=%s
    """
    params = [client_id, session_id]
    if subject_id:
        query += " AND links.subject_id=%s"
        params.append(subject_id)
    query += " ORDER BY subjects.ordinal ASC, links.created_at ASC"
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    items = []
    for row in rows:
        item = video_subject_asset_to_dict(row[:7])
        item.update({
            "public_url": row[7],
            "local_path": row[8],
            "source_asset_id": row[9],
            "asset_label": row[10],
            "operation_prompt": row[11],
        })
        items.append(item)
    return items


def approve_video_subject_asset(client_id, session_id, subject_id, asset_id=None):
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT current_asset_id
        FROM video_subjects
        WHERE client_id=%s AND session_id=%s AND subject_id=%s
        FOR UPDATE
        """,
        (client_id, session_id, subject_id),
    )
    row = cur.fetchone()
    selected_asset_id = asset_id or (row[0] if row else None)
    if not selected_asset_id:
        conn.rollback()
        cur.close()
        conn.close()
        return None
    cur.execute(
        """
        SELECT 1 FROM video_subject_assets
        WHERE subject_id=%s AND asset_id=%s AND asset_role='design_sheet'
        """,
        (subject_id, selected_asset_id),
    )
    if not cur.fetchone():
        conn.rollback()
        cur.close()
        conn.close()
        return None
    cur.execute(
        "UPDATE video_subject_assets SET pipeline_eligible=(asset_id=%s) WHERE subject_id=%s AND asset_role='design_sheet'",
        (selected_asset_id, subject_id),
    )
    cur.execute(
        """
        UPDATE video_subjects
        SET current_asset_id=%s, approved_asset_id=%s, status='approved', updated_at=%s
        WHERE client_id=%s AND session_id=%s AND subject_id=%s
        RETURNING subject_id, session_id, client_id, ordinal, display_name,
                  subject_type, story_role, spec_json, status, current_asset_id,
                  approved_asset_id, created_at, updated_at
        """,
        (selected_asset_id, selected_asset_id, now_ts, client_id, session_id, subject_id),
    )
    subject_row = cur.fetchone()
    cur.execute(
        """
        UPDATE video_projects
        SET active_subject_id=%s, status='subject_ready', updated_at=%s
        WHERE client_id=%s AND session_id=%s
        """,
        (subject_id, now_ts, client_id, session_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return video_subject_to_dict(subject_row)


def video_keyframe_to_dict(row):
    if not row:
        return None
    (
        keyframe_id,
        session_id,
        client_id,
        ordinal,
        source_beat_index,
        source_beat_fingerprint,
        time_range,
        purpose,
        script_content,
        script_visual,
        subject_ids_json,
        plan_json,
        status,
        current_asset_id,
        approved_asset_id,
        created_at,
        updated_at,
    ) = row
    return {
        "keyframe_id": keyframe_id,
        "session_id": session_id,
        "client_id": client_id,
        "ordinal": ordinal,
        "source_beat_index": source_beat_index,
        "source_beat_fingerprint": source_beat_fingerprint,
        "time_range": time_range,
        "purpose": purpose,
        "script_content": script_content,
        "script_visual": script_visual,
        "subject_ids": _json_list(subject_ids_json),
        "plan": _json_object(plan_json),
        "status": status,
        "current_asset_id": current_asset_id,
        "approved_asset_id": approved_asset_id,
        "created_at": created_at,
        "updated_at": updated_at,
    }


_VIDEO_KEYFRAME_SELECT = """
    SELECT keyframe_id, session_id, client_id, ordinal, source_beat_index,
           source_beat_fingerprint, time_range, purpose, script_content,
           script_visual, subject_ids_json, plan_json, status,
           current_asset_id, approved_asset_id, created_at, updated_at
    FROM video_keyframes
"""


def list_video_keyframes(client_id, session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        _VIDEO_KEYFRAME_SELECT + " WHERE client_id=%s AND session_id=%s ORDER BY ordinal ASC",
        (client_id, session_id),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [video_keyframe_to_dict(row) for row in rows]


def get_video_keyframe(client_id, session_id, keyframe_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        _VIDEO_KEYFRAME_SELECT + " WHERE client_id=%s AND session_id=%s AND keyframe_id=%s",
        (client_id, session_id, keyframe_id),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return video_keyframe_to_dict(row)


def replace_video_keyframe_plan(client_id, session_id, keyframes):
    """Persist the first immutable script-bound keyframe plan for a session."""
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"keyframes:{session_id}",))
    cur.execute(
        _VIDEO_KEYFRAME_SELECT + " WHERE client_id=%s AND session_id=%s ORDER BY ordinal ASC",
        (client_id, session_id),
    )
    existing_rows = cur.fetchall()
    if existing_rows:
        conn.commit()
        cur.close()
        conn.close()
        return [video_keyframe_to_dict(row) for row in existing_rows]
    for ordinal, item in enumerate(keyframes or [], start=1):
        keyframe_id = f"keyframe_{uuid.uuid4().hex}"
        cur.execute(
            """
            INSERT INTO video_keyframes (
                keyframe_id, session_id, client_id, ordinal, source_beat_index,
                source_beat_fingerprint, time_range, purpose, script_content,
                script_visual, subject_ids_json, plan_json, status,
                created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'planned', %s, %s)
            """,
            (
                keyframe_id,
                session_id,
                client_id,
                ordinal,
                int(item.get("source_beat_index") or ordinal),
                str(item.get("source_beat_fingerprint") or "")[:64],
                item.get("time_range"),
                item.get("purpose"),
                str(item.get("script_content") or ""),
                item.get("script_visual"),
                json.dumps(item.get("subject_ids") or [], ensure_ascii=False, default=str),
                json.dumps(item.get("plan") or {}, ensure_ascii=False, default=str),
                now_ts,
                now_ts,
            ),
        )
    cur.execute(
        """
        UPDATE video_projects
        SET stage='keyframe_material', status='keyframes_planned', updated_at=%s
        WHERE client_id=%s AND session_id=%s
        """,
        (now_ts, client_id, session_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return list_video_keyframes(client_id, session_id)


def video_keyframe_asset_to_dict(row):
    if not row:
        return None
    keyframe_id, asset_id, version, pipeline_eligible, inspection_json, source_subject_ids_json, created_at = row
    return {
        "keyframe_id": keyframe_id,
        "asset_id": asset_id,
        "version": version,
        "pipeline_eligible": bool(pipeline_eligible),
        "inspection": _json_object(inspection_json),
        "source_subject_ids": _json_list(source_subject_ids_json),
        "created_at": created_at,
    }


def attach_video_keyframe_asset(
    client_id,
    session_id,
    keyframe_id,
    asset_id,
    *,
    inspection=None,
    source_subject_ids=None,
):
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (keyframe_id,))
    cur.execute(
        "SELECT 1 FROM video_keyframes WHERE client_id=%s AND session_id=%s AND keyframe_id=%s FOR UPDATE",
        (client_id, session_id, keyframe_id),
    )
    if not cur.fetchone():
        conn.rollback()
        cur.close()
        conn.close()
        return None
    cur.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM video_keyframe_assets WHERE keyframe_id=%s",
        (keyframe_id,),
    )
    version = int(cur.fetchone()[0] or 1)
    cur.execute(
        """
        INSERT INTO video_keyframe_assets (
            keyframe_id, asset_id, version, pipeline_eligible, inspection_json,
            source_subject_ids_json, created_at
        )
        VALUES (%s, %s, %s, FALSE, %s, %s, %s)
        RETURNING keyframe_id, asset_id, version, pipeline_eligible,
                  inspection_json, source_subject_ids_json, created_at
        """,
        (
            keyframe_id,
            asset_id,
            version,
            json.dumps(inspection or {}, ensure_ascii=False, default=str),
            json.dumps(source_subject_ids or [], ensure_ascii=False, default=str),
            now_ts,
        ),
    )
    row = cur.fetchone()
    cur.execute(
        """
        UPDATE video_keyframes
        SET current_asset_id=%s, status='awaiting_review', updated_at=%s
        WHERE keyframe_id=%s
        """,
        (asset_id, now_ts, keyframe_id),
    )
    cur.execute(
        """
        UPDATE video_projects
        SET stage='keyframe_material', status='awaiting_keyframe_review', updated_at=%s
        WHERE client_id=%s AND session_id=%s
        """,
        (now_ts, client_id, session_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return video_keyframe_asset_to_dict(row)


def list_video_keyframe_assets(client_id, session_id, keyframe_id=None):
    conn = get_conn()
    cur = conn.cursor()
    query = """
        SELECT links.keyframe_id, links.asset_id, links.version,
               links.pipeline_eligible, links.inspection_json,
               links.source_subject_ids_json, links.created_at,
               assets.public_url, assets.local_path, assets.source_asset_id,
               assets.asset_label, assets.operation_prompt
        FROM video_keyframe_assets AS links
        JOIN video_keyframes AS frames ON frames.keyframe_id=links.keyframe_id
        JOIN chat_assets AS assets ON assets.asset_id=links.asset_id
        WHERE frames.client_id=%s AND frames.session_id=%s
    """
    params = [client_id, session_id]
    if keyframe_id:
        query += " AND links.keyframe_id=%s"
        params.append(keyframe_id)
    query += " ORDER BY frames.ordinal ASC, links.created_at ASC"
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    items = []
    for row in rows:
        item = video_keyframe_asset_to_dict(row[:7])
        item.update({
            "public_url": row[7],
            "local_path": row[8],
            "source_asset_id": row[9],
            "asset_label": row[10],
            "operation_prompt": row[11],
        })
        items.append(item)
    return items


def update_video_keyframe_asset_inspection(
    client_id,
    session_id,
    keyframe_id,
    asset_id,
    inspection,
):
    """Replace the inspection report for one existing keyframe version."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE video_keyframe_assets AS links
        SET inspection_json=%s
        FROM video_keyframes AS frames
        WHERE links.keyframe_id=frames.keyframe_id
          AND frames.client_id=%s
          AND frames.session_id=%s
          AND links.keyframe_id=%s
          AND links.asset_id=%s
        RETURNING links.keyframe_id, links.asset_id, links.version,
                  links.pipeline_eligible, links.inspection_json,
                  links.source_subject_ids_json, links.created_at
        """,
        (
            json.dumps(inspection or {}, ensure_ascii=False, default=str),
            client_id,
            session_id,
            keyframe_id,
            asset_id,
        ),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return video_keyframe_asset_to_dict(row)


def approve_video_keyframe_asset(client_id, session_id, keyframe_id, asset_id=None):
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT current_asset_id FROM video_keyframes
        WHERE client_id=%s AND session_id=%s AND keyframe_id=%s FOR UPDATE
        """,
        (client_id, session_id, keyframe_id),
    )
    row = cur.fetchone()
    selected_asset_id = asset_id or (row[0] if row else None)
    if not selected_asset_id:
        conn.rollback()
        cur.close()
        conn.close()
        return None
    cur.execute(
        "SELECT 1 FROM video_keyframe_assets WHERE keyframe_id=%s AND asset_id=%s",
        (keyframe_id, selected_asset_id),
    )
    if not cur.fetchone():
        conn.rollback()
        cur.close()
        conn.close()
        return None
    cur.execute(
        "UPDATE video_keyframe_assets SET pipeline_eligible=(asset_id=%s) WHERE keyframe_id=%s",
        (selected_asset_id, keyframe_id),
    )
    cur.execute(
        """
        UPDATE video_keyframes
        SET current_asset_id=%s, approved_asset_id=%s, status='approved', updated_at=%s
        WHERE client_id=%s AND session_id=%s AND keyframe_id=%s
        RETURNING keyframe_id, session_id, client_id, ordinal, source_beat_index,
                  source_beat_fingerprint, time_range, purpose, script_content,
                  script_visual, subject_ids_json, plan_json, status,
                  current_asset_id, approved_asset_id, created_at, updated_at
        """,
        (selected_asset_id, selected_asset_id, now_ts, client_id, session_id, keyframe_id),
    )
    keyframe_row = cur.fetchone()
    cur.execute(
        """
        SELECT COUNT(*), COUNT(*) FILTER (WHERE approved_asset_id IS NOT NULL)
        FROM video_keyframes WHERE client_id=%s AND session_id=%s
        """,
        (client_id, session_id),
    )
    total, approved = cur.fetchone()
    project_status = "keyframes_ready" if total and total == approved else "awaiting_keyframe_review"
    cur.execute(
        "UPDATE video_projects SET status=%s, updated_at=%s WHERE client_id=%s AND session_id=%s",
        (project_status, now_ts, client_id, session_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return video_keyframe_to_dict(keyframe_row)


def add_chat_message(client_id, session_id, role, content, model=None, message_type="message"):
    if role not in ("system", "user", "assistant", "tool"):
        role = "user"
    if message_type not in ("message", "interrupted"):
        message_type = "message"
    content = content if isinstance(content, str) else str(content or "")
    now_ts = int(time.time())
    message_id = f"msg_{uuid.uuid4().hex}"
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT title FROM chat_sessions
        WHERE client_id=%s AND session_id=%s AND deleted_at IS NULL
        FOR UPDATE
        """,
        (client_id, session_id),
    )
    row = cur.fetchone()
    if not row:
        conn.rollback()
        cur.close()
        conn.close()
        return None
    current_title = row[0]
    cur.execute(
        """
        INSERT INTO chat_messages (message_id, session_id, client_id, role, content, model, message_type, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING message_id, session_id, client_id, role, content, model, message_type, sequence_no, created_at
        """,
        (message_id, session_id, client_id, role, content, model, message_type, now_ts),
    )
    message_row = cur.fetchone()
    next_title = current_title
    if role == "user" and current_title == "新对话":
        next_title = re.sub(r"\s+", " ", content).strip()[:32] or current_title
    cur.execute(
        "UPDATE chat_sessions SET title=%s, updated_at=%s WHERE session_id=%s",
        (next_title, now_ts, session_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return chat_message_to_dict(message_row)


def list_chat_messages(client_id, session_id, limit=None, ascending=True):
    conn = get_conn()
    cur = conn.cursor()
    order = "ASC" if ascending else "DESC"
    limit_sql = "LIMIT %s" if limit else ""
    params = [client_id, session_id]
    if limit:
        params.append(limit)
    cur.execute(
        f"""
        SELECT message_id, session_id, client_id, role, content, model, message_type, sequence_no, created_at
        FROM chat_messages
        WHERE client_id=%s AND session_id=%s
        ORDER BY sequence_no {order}
        {limit_sql}
        """,
        tuple(params),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    messages = [chat_message_to_dict(row) for row in rows]
    return list(reversed(messages)) if limit and not ascending else messages


def create_chat_job(client_id, session_id, model=None, job_id=None):
    now_ts = int(time.time())
    job_id = job_id or f"job_{uuid.uuid4().hex}"
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO chat_jobs (job_id, session_id, client_id, status, model, created_at, updated_at)
        VALUES (%s, %s, %s, 'queued', %s, %s, %s)
        RETURNING job_id, session_id, client_id, status, model, error, usage_json, created_at, started_at, completed_at, updated_at
        """,
        (job_id, session_id, client_id, model, now_ts, now_ts),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return chat_job_to_dict(row)


def update_chat_job_status(client_id, job_id, status, error=None, usage=None):
    now_ts = int(time.time())
    if status not in ("queued", "running", "completed", "failed", "canceled"):
        status = "failed"
    usage_json = json.dumps(usage, ensure_ascii=False, default=str) if usage is not None else None
    conn = get_conn()
    cur = conn.cursor()
    started_sql = ", started_at=COALESCE(started_at, %s)" if status == "running" else ""
    completed_sql = ", completed_at=%s" if status in ("completed", "failed", "canceled") else ""
    params = [status, error, usage_json, now_ts]
    if status == "running":
        params.append(now_ts)
    if status in ("completed", "failed", "canceled"):
        params.append(now_ts)
    params.extend([client_id, job_id, status])
    cur.execute(
        f"""
        UPDATE chat_jobs
        SET status=%s, error=COALESCE(%s, error), usage_json=COALESCE(%s, usage_json),
            updated_at=%s{started_sql}{completed_sql}
        WHERE client_id=%s AND job_id=%s
          AND (status <> 'canceled' OR %s = 'canceled')
        RETURNING job_id, session_id, client_id, status, model, error, usage_json, created_at, started_at, completed_at, updated_at
        """,
        tuple(params),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return chat_job_to_dict(row)


def cancel_chat_job(client_id, job_id, error="用户已停止生成"):
    now_ts = int(time.time())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE chat_jobs
        SET status='canceled', error=%s, completed_at=%s, updated_at=%s
        WHERE client_id=%s AND job_id=%s AND status IN ('queued', 'running')
        RETURNING job_id, session_id, client_id, status, model, error, usage_json,
                  created_at, started_at, completed_at, updated_at
        """,
        (error, now_ts, now_ts, client_id, job_id),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return chat_job_to_dict(row)


def add_chat_interruption_message(client_id, session_id, model=None):
    now_ts = int(time.time())
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT message_id, session_id, client_id, role, content, model, message_type, sequence_no, created_at
        FROM chat_messages
        WHERE client_id=%s AND session_id=%s
        ORDER BY sequence_no DESC
        LIMIT 1
        FOR UPDATE
        """,
        (client_id, session_id),
    )
    latest = cur.fetchone()
    if latest and latest[6] == "interrupted":
        conn.commit()
        cur.close()
        conn.close()
        return chat_message_to_dict(latest)

    message_id = f"msg_{uuid.uuid4().hex}"
    cur.execute(
        """
        INSERT INTO chat_messages (
            message_id, session_id, client_id, role, content, model, message_type, created_at
        )
        VALUES (%s, %s, %s, 'assistant', '', %s, 'interrupted', %s)
        RETURNING message_id, session_id, client_id, role, content, model, message_type, sequence_no, created_at
        """,
        (message_id, session_id, client_id, model, now_ts),
    )
    row = cur.fetchone()
    cur.execute(
        "UPDATE chat_sessions SET updated_at=%s WHERE client_id=%s AND session_id=%s",
        (now_ts, client_id, session_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return chat_message_to_dict(row)


def regenerate_chat_message(client_id, session_id, message_id, model=None, content=None, job_id=None):
    """Replace one user turn, truncate its old branch, and enqueue a fresh job atomically."""
    now_ts = int(time.time())
    job_id = job_id or f"job_{uuid.uuid4().hex}"
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT session_id
            FROM chat_sessions
            WHERE client_id=%s AND session_id=%s AND deleted_at IS NULL
            FOR UPDATE
            """,
            (client_id, session_id),
        )
        if not cur.fetchone():
            conn.rollback()
            return {"error": "session_not_found"}

        cur.execute(
            """
            SELECT job_id
            FROM chat_jobs
            WHERE client_id=%s AND session_id=%s AND status IN ('queued', 'running')
            LIMIT 1
            FOR UPDATE
            """,
            (client_id, session_id),
        )
        if cur.fetchone():
            conn.rollback()
            return {"error": "job_running"}

        cur.execute(
            """
            SELECT message_id, session_id, client_id, role, content, model, message_type, sequence_no, created_at
            FROM chat_messages
            WHERE client_id=%s AND session_id=%s AND message_id=%s
            FOR UPDATE
            """,
            (client_id, session_id, message_id),
        )
        target = cur.fetchone()
        if not target:
            conn.rollback()
            return {"error": "message_not_found"}
        if target[3] != "user" or target[6] != "message":
            conn.rollback()
            return {"error": "message_not_editable"}

        cur.execute(
            """
            SELECT message_id
            FROM chat_messages
            WHERE client_id=%s AND session_id=%s AND role='user' AND message_type='message'
            ORDER BY sequence_no DESC
            LIMIT 1
            """,
            (client_id, session_id),
        )
        latest_user = cur.fetchone()
        if not latest_user or latest_user[0] != message_id:
            conn.rollback()
            return {"error": "message_not_latest"}

        next_content = target[4] if content is None else str(content).strip()
        if not next_content:
            conn.rollback()
            return {"error": "empty_content"}

        cur.execute(
            """
            DELETE FROM chat_messages
            WHERE client_id=%s AND session_id=%s AND sequence_no > %s
            """,
            (client_id, session_id, target[7]),
        )
        cur.execute(
            """
            UPDATE chat_messages
            SET content=%s, model=%s
            WHERE client_id=%s AND session_id=%s AND message_id=%s
            RETURNING message_id, session_id, client_id, role, content, model, message_type, sequence_no, created_at
            """,
            (next_content, model or target[5], client_id, session_id, message_id),
        )
        message_row = cur.fetchone()

        cur.execute(
            """
            SELECT message_id
            FROM chat_messages
            WHERE client_id=%s AND session_id=%s AND role='user'
            ORDER BY sequence_no ASC
            LIMIT 1
            """,
            (client_id, session_id),
        )
        first_user = cur.fetchone()
        if first_user and first_user[0] == message_id and content is not None:
            title = re.sub(r"\s+", " ", next_content).strip()[:32] or "新对话"
            cur.execute(
                "UPDATE chat_sessions SET title=%s, updated_at=%s WHERE client_id=%s AND session_id=%s",
                (title, now_ts, client_id, session_id),
            )
        else:
            cur.execute(
                "UPDATE chat_sessions SET updated_at=%s WHERE client_id=%s AND session_id=%s",
                (now_ts, client_id, session_id),
            )

        cur.execute(
            """
            INSERT INTO chat_jobs (job_id, session_id, client_id, status, model, created_at, updated_at)
            VALUES (%s, %s, %s, 'queued', %s, %s, %s)
            RETURNING job_id, session_id, client_id, status, model, error, usage_json,
                      created_at, started_at, completed_at, updated_at
            """,
            (job_id, session_id, client_id, model, now_ts, now_ts),
        )
        job_row = cur.fetchone()
        conn.commit()
        return {
            "message": chat_message_to_dict(message_row),
            "job": chat_job_to_dict(job_row),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def get_chat_job(client_id, job_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT job_id, session_id, client_id, status, model, error, usage_json, created_at, started_at, completed_at, updated_at
        FROM chat_jobs
        WHERE client_id=%s AND job_id=%s
        """,
        (client_id, job_id),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return chat_job_to_dict(row)


def latest_chat_job_for_session(client_id, session_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT job_id, session_id, client_id, status, model, error, usage_json, created_at, started_at, completed_at, updated_at
        FROM chat_jobs
        WHERE client_id=%s AND session_id=%s
        ORDER BY updated_at DESC
        LIMIT 1
        """,
        (client_id, session_id),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return chat_job_to_dict(row)


def recent_chat_job_tool_names(client_id, session_id, limit=12):
    """Return tools that were actually called in the session's recent jobs."""
    if not client_id or not session_id:
        return []

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT usage_json
        FROM chat_jobs
        WHERE client_id=%s AND session_id=%s
        ORDER BY created_at DESC
        LIMIT %s
        """,
        (client_id, session_id, max(int(limit), 1)),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    names = []
    for (usage_json,) in rows:
        if not usage_json:
            continue
        try:
            usage = json.loads(usage_json)
        except Exception:
            continue
        trace = usage.get("trace") if isinstance(usage, dict) else None
        if not isinstance(trace, list):
            continue
        for event in trace:
            if not isinstance(event, dict) or event.get("type") != "ToolCall":
                continue
            name = str(event.get("name") or "").strip()
            if name and name not in names:
                names.append(name)
    return names


def rag_collection_to_dict(row):
    if not row:
        return None
    collection_id, client_id, name, description, status, file_count, chunk_count, created_at, updated_at = row
    return {
        "collection_id": collection_id,
        "client_id": client_id,
        "name": name,
        "description": description,
        "status": status,
        "file_count": file_count,
        "chunk_count": chunk_count,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def rag_document_to_dict(row):
    if not row:
        return None
    document_id, collection_id, client_id, filename, stored_path, content_type, file_ext, status, error, chunk_count, created_at, updated_at = row
    return {
        "document_id": document_id,
        "collection_id": collection_id,
        "client_id": client_id,
        "filename": filename,
        "stored_path": stored_path,
        "content_type": content_type,
        "file_ext": file_ext,
        "status": status,
        "error": error,
        "chunk_count": chunk_count,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def rag_job_to_dict(row):
    if not row:
        return None
    job_id, collection_id, document_id, client_id, status, stage, error, progress_current, progress_total, created_at, started_at, completed_at, updated_at = row
    return {
        "job_id": job_id,
        "collection_id": collection_id,
        "document_id": document_id,
        "client_id": client_id,
        "status": status,
        "stage": stage,
        "error": error,
        "progress_current": progress_current,
        "progress_total": progress_total,
        "created_at": created_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "updated_at": updated_at,
    }


def create_rag_collection(client_id, name=None, description=None, project_id=DEFAULT_PROJECT_ID):
    client_id = ensure_core_client(client_id)
    project_id = ensure_core_project(client_id, project_id)
    now_ts = _now_ts()
    collection_id = f"kb_{uuid.uuid4().hex}"
    name = (name or "未命名知识库").strip()[:120] or "未命名知识库"
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rag_collections (collection_id, client_id, project_id, name, description, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING collection_id, client_id, name, description, status, file_count, chunk_count, created_at, updated_at
        """,
        (collection_id, client_id, project_id, name, description, now_ts, now_ts),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return rag_collection_to_dict(row)


def get_rag_collection(client_id, collection_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT collection_id, client_id, name, description, status, file_count, chunk_count, created_at, updated_at
        FROM rag_collections
        WHERE client_id=%s AND collection_id=%s AND deleted_at IS NULL
        """,
        (client_id, collection_id),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return rag_collection_to_dict(row)


def list_rag_collections(client_id, limit=100):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT collection_id, client_id, name, description, status, file_count, chunk_count, created_at, updated_at
        FROM rag_collections
        WHERE client_id=%s AND deleted_at IS NULL
        ORDER BY updated_at DESC
        LIMIT %s
        """,
        (client_id, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [rag_collection_to_dict(row) for row in rows]


def create_rag_document(client_id, collection_id, filename, stored_path, content_type=None, file_ext=None):
    now_ts = _now_ts()
    document_id = f"doc_{uuid.uuid4().hex}"
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rag_documents (
            document_id, collection_id, client_id, filename, stored_path, content_type, file_ext,
            status, created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, 'queued', %s, %s)
        RETURNING document_id, collection_id, client_id, filename, stored_path, content_type, file_ext,
                  status, error, chunk_count, created_at, updated_at
        """,
        (document_id, collection_id, client_id, filename, stored_path, content_type, file_ext, now_ts, now_ts),
    )
    row = cur.fetchone()
    cur.execute(
        """
        UPDATE rag_collections
        SET file_count=file_count + 1, status='indexing', updated_at=%s
        WHERE client_id=%s AND collection_id=%s
        """,
        (now_ts, client_id, collection_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return rag_document_to_dict(row)


def list_rag_documents(client_id, collection_id, limit=100):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT document_id, collection_id, client_id, filename, stored_path, content_type, file_ext,
               status, error, chunk_count, created_at, updated_at
        FROM rag_documents
        WHERE client_id=%s AND collection_id=%s
        ORDER BY updated_at DESC
        LIMIT %s
        """,
        (client_id, collection_id, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [rag_document_to_dict(row) for row in rows]


def create_rag_job(client_id, collection_id, document_id):
    now_ts = _now_ts()
    job_id = f"ragjob_{uuid.uuid4().hex}"
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rag_jobs (job_id, collection_id, document_id, client_id, status, stage, created_at, updated_at)
        VALUES (%s, %s, %s, %s, 'queued', 'queued', %s, %s)
        RETURNING job_id, collection_id, document_id, client_id, status, stage, error,
                  progress_current, progress_total, created_at, started_at, completed_at, updated_at
        """,
        (job_id, collection_id, document_id, client_id, now_ts, now_ts),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return rag_job_to_dict(row)


def update_rag_job_status(client_id, job_id, status=None, stage=None, error=None, progress_current=None, progress_total=None):
    now_ts = _now_ts()
    status = status or "running"
    if status not in ("queued", "running", "completed", "failed"):
        status = "failed"
    stage = stage or status
    conn = get_conn()
    cur = conn.cursor()
    started_sql = ", started_at=COALESCE(started_at, %s)" if status == "running" else ""
    completed_sql = ", completed_at=%s" if status in ("completed", "failed") else ""
    params = [status, stage, error, progress_current, progress_total, now_ts]
    if status == "running":
        params.append(now_ts)
    if status in ("completed", "failed"):
        params.append(now_ts)
    params.extend([client_id, job_id])
    cur.execute(
        f"""
        UPDATE rag_jobs
        SET status=%s,
            stage=%s,
            error=COALESCE(%s, error),
            progress_current=COALESCE(%s, progress_current),
            progress_total=COALESCE(%s, progress_total),
            updated_at=%s{started_sql}{completed_sql}
        WHERE client_id=%s AND job_id=%s
        RETURNING job_id, collection_id, document_id, client_id, status, stage, error,
                  progress_current, progress_total, created_at, started_at, completed_at, updated_at
        """,
        tuple(params),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return rag_job_to_dict(row)


def get_rag_job(client_id, job_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT job_id, collection_id, document_id, client_id, status, stage, error,
               progress_current, progress_total, created_at, started_at, completed_at, updated_at
        FROM rag_jobs
        WHERE client_id=%s AND job_id=%s
        """,
        (client_id, job_id),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return rag_job_to_dict(row)


def update_rag_document_status(client_id, document_id, status, error=None, chunk_count=None):
    now_ts = _now_ts()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE rag_documents
        SET status=%s, error=%s, chunk_count=COALESCE(%s, chunk_count), updated_at=%s
        WHERE client_id=%s AND document_id=%s
        RETURNING document_id, collection_id, client_id, filename, stored_path, content_type, file_ext,
                  status, error, chunk_count, created_at, updated_at
        """,
        (status, error, chunk_count, now_ts, client_id, document_id),
    )
    row = cur.fetchone()
    if row:
        collection_id = row[1]
        cur.execute(
            """
            UPDATE rag_collections c
            SET chunk_count=(
                    SELECT COALESCE(SUM(chunk_count), 0)
                    FROM rag_documents
                    WHERE collection_id=c.collection_id AND client_id=c.client_id AND status='completed'
                ),
                status=CASE
                    WHEN EXISTS (
                        SELECT 1 FROM rag_documents
                        WHERE collection_id=c.collection_id AND client_id=c.client_id AND status IN ('queued', 'running')
                    ) THEN 'indexing'
                    WHEN EXISTS (
                        SELECT 1 FROM rag_documents
                        WHERE collection_id=c.collection_id AND client_id=c.client_id AND status='completed'
                    ) THEN 'ready'
                    ELSE 'empty'
                END,
                updated_at=%s
            WHERE c.client_id=%s AND c.collection_id=%s
            """,
            (now_ts, client_id, collection_id),
        )
    conn.commit()
    cur.close()
    conn.close()
    return rag_document_to_dict(row)


def delete_rag_chunks_for_document(client_id, document_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM rag_chunks WHERE client_id=%s AND document_id=%s", (client_id, document_id))
    deleted = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()
    return deleted


def insert_rag_chunk(client_id, collection_id, document_id, chunk_index, content, vector, metadata=None):
    now_ts = _now_ts()
    chunk_id = f"chunk_{uuid.uuid4().hex}"
    metadata_json = json.dumps(metadata or {}, ensure_ascii=False, default=str)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO rag_chunks (
            chunk_id, document_id, collection_id, client_id, chunk_index, content, embedding, metadata_json, created_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
        RETURNING chunk_id
        """,
        (chunk_id, document_id, collection_id, client_id, chunk_index, content, _vector_literal(vector), metadata_json, now_ts),
    )
    conn.commit()
    cur.close()
    conn.close()
    return chunk_id


def search_rag_chunks(client_id, query_vector, collection_id=None, limit=5):
    limit = max(1, min(int(limit or 5), 20))
    vector_value = _vector_literal(query_vector)
    params = [vector_value, client_id]
    collection_filter = ""
    if collection_id:
        collection_filter = "AND collection_id=%s"
        params.append(collection_id)
    params.extend([vector_value, limit])
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT chunk_id, document_id, collection_id, chunk_index, content,
               metadata_json, 1 - (embedding <=> %s::vector) AS score
        FROM rag_chunks
        WHERE client_id=%s {collection_filter}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        tuple(params),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    results = []
    for row in rows:
        chunk_id, document_id, collection_id, chunk_index, content, metadata_json, score = row
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except Exception:
            metadata = {}
        results.append({
            "chunk_id": chunk_id,
            "document_id": document_id,
            "collection_id": collection_id,
            "chunk_index": chunk_index,
            "content": content,
            "metadata": metadata,
            "score": float(score) if score is not None else None,
        })
    return results

# if __name__ == "__main__":
    # uid = register_user('testuser', 'mypassword')
    # print("新用户ID：", uid)
    # uid2 = login_user('testuser', 'mypassword')
    # print("登录成功，ID：", uid2)
    # api_key = add_apikey_for_user(uid)
    # print("分配新key（只展示一次）：", api_key)
    # print("用户所有密钥：", get_user_keys(uid))
    # print("余额：", get_token_balance(uid))
    # print("扣除50token:", deduct_tokens(uid, 50))
    # print("余额：", get_token_balance(uid))
    # print("充值30token:", recharge_tokens(uid, 30))
    # print("余额：", get_token_balance(uid))

ensure_usage_schema()
RAG_SCHEMA_READY = ensure_rag_schema()
