/**
 * *********************************************
 *
 * \file Auth.h
 * \brief Header of the Kalman::Net::Auth class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Net::Auth
 * \brief Verifies the scoped JWTs that the Python side issues.
 *
 * \details The tokens are those of modules/api/auth.py: HS256 over the secret in
 * `KALMAN_JWT_SECRET`, with the claims `sub`, `scopes`, `iat`, `exp`, `jti`, and
 * `iss`/`aud` when `KALMAN_JWT_ISSUER`/`KALMAN_JWT_AUDIENCE` are set. A token
 * minted by `python -m kalman.api.auth token` or `POST /token` is accepted here
 * unchanged, and a token revoked through Python is refused here, because the
 * revocation list is read from the same registry (`$KALMAN_STATE_DIR/registry.sqlite`).
 *
 * The rules are Python's, checked in the same order:
 *
 *  - no secret configured: auth is off, every call is allowed (local use);
 *  - a secret configured: a token is required, its signature, `exp` (and `nbf`,
 *    `iat`), `iss` and `aud` must hold, it must not be revoked (401 otherwise),
 *    and it must carry the scope or `admin` (403 otherwise).
 *
 * Issuing tokens and managing accounts (bcrypt, the htpasswd file, logins and
 * their lockout) stay in Python: this class only verifies. Sign() exists for
 * tests and for tools that already hold the secret.
 *
 * *********************************************
 */

#ifndef KALMAN_NET_AUTH_H
#define KALMAN_NET_AUTH_H

#include <string>
#include <vector>

namespace Kalman::Net
{
    class Registry;

    /// The verified content of a token.
    struct Claims
    {
        std::string subject;              ///< `sub`, the user
        std::vector<std::string> scopes;  ///< `scopes`
        double exp = 0.;                  ///< `exp`, seconds since the epoch
        double iat = 0.;                  ///< `iat`, when hasIat
        bool hasIat = false;
        std::string jti;                  ///< `jti`, the token's own id ("" when absent)
        std::string json;                 ///< the whole payload, as JSON text
    };

    class Auth
    {
        private:
            std::string fSecret;
            std::string fIssuer;
            std::string fAudience;
            const Registry *fRegistry = nullptr;   // the revocation list (not owned)

        public:
            /// The scopes a token may carry, as in auth.py.
            static const std::vector<std::string> &Scopes();

            /// HS256 needs a key at least as long as its output (RFC 7518, 3.2).
            static constexpr std::size_t kMinSecretBytes = 32;

            /// Configured from the environment: KALMAN_JWT_SECRET, _ISSUER, _AUDIENCE.
            Auth();
            Auth(std::string secret, std::string issuer = "", std::string audience = "");

            /// Where revocations are read from; nullptr reads none (and, with auth
            /// on, refuses every token: fail-closed, as auth.py does).
            void SetRegistry(const Registry *registry) { fRegistry = registry; }

            /// True when a secret is configured.
            bool Enabled() const { return !fSecret.empty(); }
            const std::string &Secret() const { return fSecret; }

            /// Verify signature, expiry, issuer, audience and revocation.
            /// \throws Kalman::Net::Error 401 on any failure.
            Claims Verify(const std::string &token) const;

            /// Verify \p token and check it carries \p scope (or admin). No-op when
            /// auth is off. \throws Kalman::Net::Error 401 (token) or 403 (scope);
            /// 400 for a scope name that does not exist.
            void Authorize(const std::string &token, const std::string &scope) const;

            /// The token's user, or "" when auth is off.
            std::string Identity(const std::string &token) const;

            /// A HS256 JWT over \p payloadJson (a JSON object, as text).
            static std::string Sign(const std::string &payloadJson, const std::string &secret);
    };
}

#endif // KALMAN_NET_AUTH_H
