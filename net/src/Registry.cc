/**
 * \file Registry.cc
 * \brief The shared registry (modules/api/store.py) from C++.
 */
#include "Registry.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <sys/stat.h>

#if KALMAN_NET_WITH_SQLITE
#include <sqlite3.h>
#endif

namespace fs = std::filesystem;

namespace Kalman::Net
{
    namespace
    {
        std::string DefaultPath()
        {
            std::string dir;
            if (const char *s = std::getenv("KALMAN_STATE_DIR"); s && *s) dir = s;
            else
            {
                const char *home = std::getenv("HOME");
                dir = std::string(home ? home : ".") + "/.kalman/state";
            }
            if (!dir.empty() && dir[0] == '~')
            {
                const char *home = std::getenv("HOME");
                dir = std::string(home ? home : ".") + dir.substr(1);
            }
            return (fs::absolute(fs::path(dir)) / "registry.sqlite").string();
        }

        double Now()
        {
            using namespace std::chrono;
            return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
        }

#if KALMAN_NET_WITH_SQLITE
        // store.py's schema, so a file this side creates is the one Python expects.
        const char *kSchema = R"SQL(
CREATE TABLE IF NOT EXISTS models (
    name    TEXT PRIMARY KEY, spec TEXT NOT NULL, info TEXT NOT NULL, owner TEXT,
    created REAL NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS plugins (
    name    TEXT PRIMARY KEY, path TEXT NOT NULL, models TEXT NOT NULL, owner TEXT,
    loaded  REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sources (
    name    TEXT PRIMARY KEY, spec TEXT NOT NULL, owner TEXT,
    created REAL NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS measurements (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, row TEXT NOT NULL,
    t REAL, owner TEXT, received REAL NOT NULL);
CREATE INDEX IF NOT EXISTS measurements_by_source ON measurements(source, seq);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, type TEXT NOT NULL, payload TEXT NOT NULL,
    ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS revoked_tokens (
    jti TEXT PRIMARY KEY, sub TEXT, exp REAL, by TEXT, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS revoked_users (
    sub TEXT PRIMARY KEY, before REAL NOT NULL, by TEXT, at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS login_failures (
    user TEXT NOT NULL, client TEXT NOT NULL, at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS login_failures_by_user ON login_failures(user, client, at);
)SQL";

        // One connection per call, as store.py: they never cross threads.
        class Db
        {
            public:
                sqlite3 *fDb = nullptr;

                Db(const std::string &path, bool create)
                {
                    if (create)
                    {
                        fs::create_directories(fs::path(path).parent_path());
                        const bool fresh = !fs::exists(path);
                        Open(path, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE);
                        Exec("PRAGMA journal_mode=WAL");
                        Exec(kSchema);
                        if (fresh) ::chmod(path.c_str(), 0600);
                    }
                    else Open(path, SQLITE_OPEN_READWRITE);
                }
                ~Db() { if (fDb) sqlite3_close(fDb); }

                void Open(const std::string &path, int flags)
                {
                    if (sqlite3_open_v2(path.c_str(), &fDb, flags, nullptr) != SQLITE_OK)
                    {
                        const std::string why = fDb ? sqlite3_errmsg(fDb) : "out of memory";
                        throw std::runtime_error("registry " + path + ": " + why);
                    }
                    sqlite3_busy_timeout(fDb, 30000);
                }

                void Exec(const char *sql)
                {
                    char *err = nullptr;
                    if (sqlite3_exec(fDb, sql, nullptr, nullptr, &err) != SQLITE_OK)
                    {
                        const std::string why = err ? err : "sqlite error";
                        sqlite3_free(err);
                        throw std::runtime_error("registry: " + why);
                    }
                }
        };

        class Stmt
        {
            public:
                sqlite3_stmt *fStmt = nullptr;
                sqlite3 *fDb;

                Stmt(Db &db, const char *sql) : fDb(db.fDb)
                {
                    if (sqlite3_prepare_v2(fDb, sql, -1, &fStmt, nullptr) != SQLITE_OK)
                        throw std::runtime_error(std::string("registry: ") + sqlite3_errmsg(fDb));
                }
                ~Stmt() { sqlite3_finalize(fStmt); }

                Stmt &Bind(int i, const std::string &s)
                {
                    sqlite3_bind_text(fStmt, i, s.c_str(), static_cast<int>(s.size()), SQLITE_TRANSIENT);
                    return *this;
                }
                Stmt &Bind(int i, long long v) { sqlite3_bind_int64(fStmt, i, v); return *this; }
                Stmt &Bind(int i, double v) { sqlite3_bind_double(fStmt, i, v); return *this; }
                Stmt &BindNull(int i) { sqlite3_bind_null(fStmt, i); return *this; }

                bool Step()
                {
                    const int rc = sqlite3_step(fStmt);
                    if (rc == SQLITE_ROW) return true;
                    if (rc == SQLITE_DONE) return false;
                    throw std::runtime_error(std::string("registry: ") + sqlite3_errmsg(fDb));
                }

                bool IsNull(int c) const { return sqlite3_column_type(fStmt, c) == SQLITE_NULL; }
                std::string Text(int c) const
                {
                    const unsigned char *s = sqlite3_column_text(fStmt, c);
                    return s ? reinterpret_cast<const char *>(s) : "";
                }
                json TextOrNull(int c) const { return IsNull(c) ? json(nullptr) : json(Text(c)); }
                json Real(int c) const { return IsNull(c) ? json(nullptr) : json(sqlite3_column_double(fStmt, c)); }
                long long Int(int c) const { return sqlite3_column_int64(fStmt, c); }
                json Json(int c) const { return json::parse(Text(c)); }
        };

        bool MissingTable(const std::runtime_error &e)
        {
            return std::string(e.what()).find("no such table") != std::string::npos;
        }

        // A read on a registry nobody has created yet is an empty answer, as in
        // Python (which would create it empty first).
        template <typename F, typename R>
        R Read(const std::string &path, R empty, F &&body)
        {
            if (!fs::exists(path)) return empty;
            try
            {
                Db db(path, false);
                return body(db);
            }
            catch (const std::runtime_error &e)
            {
                if (MissingTable(e)) return empty;
                throw;
            }
        }
#endif
    }

    Registry::Registry(std::string path) : fPath(path.empty() ? DefaultPath() : std::move(path)) {}

    bool Registry::Available() { return KALMAN_NET_WITH_SQLITE != 0; }

    bool Registry::Exists() const { return fs::exists(fPath); }

#if KALMAN_NET_WITH_SQLITE
    namespace
    {
        json ModelRow(Stmt &s)
        {
            // name, spec, info, owner, created, updated
            json out = {{"name", s.Text(0)}, {"spec", s.Json(1)}};
            const json info = s.Json(2);
            if (info.is_object())
                for (auto it = info.begin(); it != info.end(); ++it) out[it.key()] = it.value();
            out["owner"] = s.TextOrNull(3);
            out["created"] = s.Real(4);
            out["updated"] = s.Real(5);
            return out;
        }

        json SourceRow(Stmt &s)
        {
            // name, spec, owner, created, updated
            json out = {{"name", s.Text(0)}};
            const json spec = s.Json(1);
            if (spec.is_object())
                for (auto it = spec.begin(); it != spec.end(); ++it) out[it.key()] = it.value();
            out["owner"] = s.TextOrNull(2);
            out["created"] = s.Real(3);
            out["updated"] = s.Real(4);
            return out;
        }
    }

    std::optional<json> Registry::GetModel(const std::string &name) const
    {
        return Read(fPath, std::optional<json>(), [&](Db &db) -> std::optional<json> {
            Stmt s(db, "SELECT name, spec, info, owner, created, updated FROM models WHERE name=?");
            s.Bind(1, name);
            if (!s.Step()) return std::nullopt;
            return ModelRow(s);
        });
    }

    std::vector<std::string> Registry::ModelNames() const
    {
        return Read(fPath, std::vector<std::string>(), [&](Db &db) {
            std::vector<std::string> out;
            Stmt s(db, "SELECT name FROM models ORDER BY name");
            while (s.Step()) out.push_back(s.Text(0));
            return out;
        });
    }

    json Registry::ListModels() const
    {
        return Read(fPath, json::array(), [&](Db &db) {
            json out = json::array();
            Stmt s(db, "SELECT name, spec, info, owner, created, updated FROM models ORDER BY name");
            while (s.Step()) out.push_back(ModelRow(s));
            return out;
        });
    }

    json Registry::ListPlugins() const
    {
        return Read(fPath, json::array(), [&](Db &db) {
            json out = json::array();
            Stmt s(db, "SELECT name, path, models, owner, loaded FROM plugins ORDER BY name");
            while (s.Step())
                out.push_back({{"name", s.Text(0)}, {"path", s.Text(1)}, {"models", s.Json(2)},
                               {"owner", s.TextOrNull(3)}, {"loaded", s.Real(4)}});
            return out;
        });
    }

    bool Registry::PutPlugin(const std::string &name, const std::string &path,
                             const std::vector<std::string> &models, const json &owner) const
    {
        std::vector<std::string> sorted = models;
        std::sort(sorted.begin(), sorted.end());
        Db db(fPath, true);
        db.Exec("BEGIN IMMEDIATE");
        try
        {
            {
                Stmt s(db, "SELECT models FROM plugins WHERE name=?");
                s.Bind(1, name);
                if (s.Step())
                {
                    std::vector<std::string> had = s.Json(0).get<std::vector<std::string>>();
                    std::sort(had.begin(), had.end());
                    if (had == sorted) { db.Exec("COMMIT"); return false; }
                }
            }
            const double now = Now();
            {
                Stmt s(db, "INSERT OR REPLACE INTO plugins VALUES (?,?,?,?,?)");
                s.Bind(1, name).Bind(2, path).Bind(3, json(sorted).dump()).Bind(5, now);
                if (owner.is_string()) s.Bind(4, owner.get<std::string>()); else s.BindNull(4);
                s.Step();
            }
            {
                const json payload = {{"name", name}, {"models", sorted}, {"owner", owner}};
                Stmt s(db, "INSERT INTO events(type, payload, ts) VALUES (?,?,?)");
                s.Bind(1, std::string("plugin.loaded")).Bind(2, payload.dump()).Bind(3, now);
                s.Step();
            }
            const long long eid = sqlite3_last_insert_rowid(db.fDb);
            long long keep = 10000;
            if (const char *e = std::getenv("KALMAN_MAX_EVENTS")) keep = std::atoll(e);
            if (keep > 0)
            {
                Stmt s(db, "DELETE FROM events WHERE id <= ?");
                s.Bind(1, eid - keep);
                s.Step();
            }
            db.Exec("COMMIT");
        }
        catch (...)
        {
            sqlite3_exec(db.fDb, "ROLLBACK", nullptr, nullptr, nullptr);
            throw;
        }
        return true;
    }

    std::optional<json> Registry::GetSource(const std::string &name) const
    {
        return Read(fPath, std::optional<json>(), [&](Db &db) -> std::optional<json> {
            Stmt s(db, "SELECT name, spec, owner, created, updated FROM sources WHERE name=?");
            s.Bind(1, name);
            if (!s.Step()) return std::nullopt;
            return SourceRow(s);
        });
    }

    json Registry::ListSources() const
    {
        return Read(fPath, json::array(), [&](Db &db) {
            json out = json::array();
            Stmt s(db, "SELECT name, spec, owner, created, updated FROM sources ORDER BY name");
            while (s.Step())
            {
                json rec = SourceRow(s);
                if (rec.value("kind", "") == "push")
                {
                    Stmt c(db, "SELECT COUNT(*), MAX(seq) FROM measurements WHERE source=?");
                    c.Bind(1, rec["name"].get<std::string>());
                    c.Step();
                    rec["rows"] = c.Int(0);
                    rec["last"] = c.IsNull(1) ? 0 : c.Int(1);
                }
                out.push_back(rec);
            }
            return out;
        });
    }

    json Registry::ReadMeasurements(const std::string &source, long long since, long long limit) const
    {
        return Read(fPath, json::array(), [&](Db &db) {
            json out = json::array();
            Stmt s(db, "SELECT seq, row, t, received FROM measurements "
                       "WHERE source=? AND seq>? ORDER BY seq LIMIT ?");
            s.Bind(1, source).Bind(2, since).Bind(3, limit);
            while (s.Step())
                out.push_back({{"seq", s.Int(0)}, {"row", s.Json(1)}, {"t", s.Real(2)}, {"received", s.Real(3)}});
            return out;
        });
    }

    json Registry::Events(long long since, long long limit) const
    {
        return Read(fPath, json::array(), [&](Db &db) {
            json out = json::array();
            Stmt s(db, "SELECT id, type, payload, ts FROM events WHERE id>? ORDER BY id LIMIT ?");
            s.Bind(1, since).Bind(2, limit);
            while (s.Step())
                out.push_back({{"id", s.Int(0)}, {"type", s.Text(1)}, {"payload", s.Json(2)}, {"ts", s.Real(3)}});
            return out;
        });
    }

    long long Registry::LastEvent() const
    {
        return Read(fPath, 0LL, [&](Db &db) {
            Stmt s(db, "SELECT COALESCE(MAX(id), 0) FROM events");
            s.Step();
            return s.Int(0);
        });
    }

    bool Registry::IsRevoked(const std::string &jti, const std::string &sub, const double *iat) const
    {
        return Read(fPath, false, [&](Db &db) {
            if (!jti.empty())
            {
                Stmt s(db, "SELECT 1 FROM revoked_tokens WHERE jti=?");
                s.Bind(1, jti);
                if (s.Step()) return true;
            }
            Stmt s(db, "SELECT before FROM revoked_users WHERE sub=?");
            s.Bind(1, sub);
            if (s.Step())
            {
                // A token with no iat cannot be shown to be newer than the cutoff.
                const double before = sqlite3_column_double(s.fStmt, 0);
                if (!iat || *iat <= before) return true;
            }
            return false;
        });
    }
#else
    std::optional<json> Registry::GetModel(const std::string &) const { return std::nullopt; }
    std::vector<std::string> Registry::ModelNames() const { return {}; }
    json Registry::ListModels() const { return json::array(); }
    json Registry::ListPlugins() const { return json::array(); }
    bool Registry::PutPlugin(const std::string &, const std::string &, const std::vector<std::string> &,
                             const json &) const { return false; }
    std::optional<json> Registry::GetSource(const std::string &) const { return std::nullopt; }
    json Registry::ListSources() const { return json::array(); }
    json Registry::ReadMeasurements(const std::string &, long long, long long) const { return json::array(); }
    json Registry::Events(long long, long long) const { return json::array(); }
    long long Registry::LastEvent() const { return 0; }
    bool Registry::IsRevoked(const std::string &, const std::string &, const double *) const
    {
        throw std::runtime_error("this libKalmanNet was built without SQLite: no revocation list");
    }
#endif
}
