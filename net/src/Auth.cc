/**
 * \file Auth.cc
 * \brief HS256 JWT verification, compatible with modules/api/auth.py (PyJWT).
 */
#include "Kalman/Net/Auth.h"
#include "Kalman/Net/Error.h"

#include "Registry.h"

#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <stdexcept>

namespace Kalman::Net
{
    namespace
    {
        std::string Env(const char *name)
        {
            const char *s = std::getenv(name);
            return s ? std::string(s) : std::string();
        }

        const std::string kAlphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

        std::string Base64UrlEncode(const unsigned char *data, std::size_t size)
        {
            std::string out;
            out.reserve((size + 2) / 3 * 4);
            std::size_t i = 0;
            for (; i + 2 < size; i += 3)
            {
                const unsigned v = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
                out += kAlphabet[(v >> 18) & 63];
                out += kAlphabet[(v >> 12) & 63];
                out += kAlphabet[(v >> 6) & 63];
                out += kAlphabet[v & 63];
            }
            if (i + 1 == size)
            {
                const unsigned v = data[i] << 16;
                out += kAlphabet[(v >> 18) & 63];
                out += kAlphabet[(v >> 12) & 63];
            }
            else if (i + 2 == size)
            {
                const unsigned v = (data[i] << 16) | (data[i + 1] << 8);
                out += kAlphabet[(v >> 18) & 63];
                out += kAlphabet[(v >> 12) & 63];
                out += kAlphabet[(v >> 6) & 63];
            }
            return out;   // unpadded, as JWS wants
        }

        bool Base64UrlDecode(const std::string &in, std::string &out)
        {
            out.clear();
            unsigned value = 0;
            int bits = 0;
            for (char c : in)
            {
                if (c == '=') break;
                int d;
                if (c >= 'A' && c <= 'Z') d = c - 'A';
                else if (c >= 'a' && c <= 'z') d = c - 'a' + 26;
                else if (c >= '0' && c <= '9') d = c - '0' + 52;
                else if (c == '-' || c == '+') d = 62;
                else if (c == '_' || c == '/') d = 63;
                else return false;
                value = (value << 6) | static_cast<unsigned>(d);
                bits += 6;
                if (bits >= 8)
                {
                    bits -= 8;
                    out += static_cast<char>((value >> bits) & 0xFF);
                }
            }
            return true;
        }

        std::string Hmac(const std::string &secret, const std::string &message)
        {
            unsigned char mac[EVP_MAX_MD_SIZE];
            unsigned int len = 0;
            if (!HMAC(EVP_sha256(), secret.data(), static_cast<int>(secret.size()),
                      reinterpret_cast<const unsigned char *>(message.data()), message.size(), mac, &len))
                throw Error(401, "invalid or expired token: HMAC failed.");
            return Base64UrlEncode(mac, len);
        }

        [[noreturn]] void Invalid(const std::string &why)
        {
            throw Error(401, "invalid or expired token: " + why + ".");
        }

        double Now()
        {
            using namespace std::chrono;
            return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
        }
    }

    const std::vector<std::string> &Auth::Scopes()
    {
        static const std::vector<std::string> scopes = {"read", "send_measurement", "write_models",
                                                        "load_plugins", "admin"};
        return scopes;
    }

    Auth::Auth()
        : fSecret(Env("KALMAN_JWT_SECRET")), fIssuer(Env("KALMAN_JWT_ISSUER")),
          fAudience(Env("KALMAN_JWT_AUDIENCE"))
    {
    }

    Auth::Auth(std::string secret, std::string issuer, std::string audience)
        : fSecret(std::move(secret)), fIssuer(std::move(issuer)), fAudience(std::move(audience))
    {
    }

    std::string Auth::Sign(const std::string &payloadJson, const std::string &secret)
    {
        const json payload = json::parse(payloadJson);
        if (!payload.is_object()) throw std::invalid_argument("a JWT payload is a JSON object");
        const std::string header = R"({"alg":"HS256","typ":"JWT"})";
        const std::string body = payload.dump();
        const std::string signing =
            Base64UrlEncode(reinterpret_cast<const unsigned char *>(header.data()), header.size()) + "."
            + Base64UrlEncode(reinterpret_cast<const unsigned char *>(body.data()), body.size());
        return signing + "." + Hmac(secret, signing);
    }

    Claims Auth::Verify(const std::string &token) const
    {
        if (!Enabled()) throw Error(401, "authentication is off; there is no token to verify.");
        if (token.empty()) throw Error(401, "a bearer token is required.");

        // The names in the messages are PyJWT's exception names, so a client sees
        // the same reason from either server.
        const std::size_t a = token.find('.');
        const std::size_t b = a == std::string::npos ? a : token.find('.', a + 1);
        if (a == std::string::npos || b == std::string::npos || token.find('.', b + 1) != std::string::npos)
            Invalid("DecodeError");

        std::string headerText, payloadText;
        if (!Base64UrlDecode(token.substr(0, a), headerText)
            || !Base64UrlDecode(token.substr(a + 1, b - a - 1), payloadText))
            Invalid("DecodeError");

        json header, payload;
        try
        {
            header = json::parse(headerText);
            payload = json::parse(payloadText);
        }
        catch (const std::exception &) { Invalid("DecodeError"); }
        if (!header.is_object() || !payload.is_object()) Invalid("DecodeError");
        if (header.value("alg", "") != "HS256") Invalid("InvalidAlgorithmError");

        // Constant-time comparison of the signature.
        const std::string expected = Hmac(fSecret, token.substr(0, b));
        const std::string given = token.substr(b + 1);
        if (given.size() != expected.size()
            || CRYPTO_memcmp(given.data(), expected.data(), expected.size()) != 0)
            Invalid("InvalidSignatureError");

        // options={"require": ["exp", "sub"]}
        if (!payload.contains("exp") || !payload.contains("sub")) Invalid("MissingRequiredClaimError");
        if (!payload["exp"].is_number()) Invalid("DecodeError");
        if (!payload["sub"].is_string()) Invalid("InvalidSubjectError");

        const double now = Now();
        Claims c;
        c.subject = payload["sub"].get<std::string>();
        c.exp = payload["exp"].get<double>();
        if (c.exp <= now) Invalid("ExpiredSignatureError");
        if (payload.contains("iat"))
        {
            if (!payload["iat"].is_number()) Invalid("InvalidIssuedAtError");
            c.iat = payload["iat"].get<double>();
            c.hasIat = true;
            if (c.iat > now) Invalid("ImmatureSignatureError");
        }
        if (payload.contains("nbf"))
        {
            if (!payload["nbf"].is_number()) Invalid("DecodeError");
            if (payload["nbf"].get<double>() > now) Invalid("ImmatureSignatureError");
        }

        // Issuer: required and equal when configured.
        if (!fIssuer.empty())
        {
            if (!payload.contains("iss")) Invalid("MissingRequiredClaimError");
            if (!payload["iss"].is_string() || payload["iss"].get<std::string>() != fIssuer)
                Invalid("InvalidIssuerError");
        }
        // Audience: required and matching when configured; a token that names an
        // audience is refused by a server that expects none (PyJWT's rule).
        if (!fAudience.empty())
        {
            if (!payload.contains("aud")) Invalid("MissingRequiredClaimError");
            const json &aud = payload["aud"];
            bool match = false;
            if (aud.is_string()) match = aud.get<std::string>() == fAudience;
            else if (aud.is_array())
                for (const auto &x : aud) match = match || (x.is_string() && x.get<std::string>() == fAudience);
            if (!match) Invalid("InvalidAudienceError");
        }
        else if (payload.contains("aud")) Invalid("InvalidAudienceError");

        if (payload.contains("scopes") && payload["scopes"].is_array())
            for (const auto &s : payload["scopes"])
                if (s.is_string()) c.scopes.push_back(s.get<std::string>());
        if (payload.contains("jti") && payload["jti"].is_string()) c.jti = payload["jti"].get<std::string>();
        c.json = payload.dump();

        // The revocation list -- fail-closed: no list, no entry.
        bool revoked = false;
        try
        {
            if (!fRegistry) throw std::runtime_error("no registry");
            revoked = fRegistry->IsRevoked(c.jti, c.subject, c.hasIat ? &c.iat : nullptr);
        }
        catch (const std::exception &e)
        {
            throw Error(401, std::string("cannot read the revocation list: ") + e.what() + ".");
        }
        if (revoked) throw Error(401, "this token was revoked.");
        return c;
    }

    void Auth::Authorize(const std::string &token, const std::string &scope) const
    {
        const auto &all = Scopes();
        if (std::find(all.begin(), all.end(), scope) == all.end())
            throw Error(400, "unknown scope '" + scope + "'.");
        if (!Enabled()) return;                          // auth disabled -- local use
        if (token.empty()) throw Error(401, "a bearer token with scope '" + scope + "' is required.");
        const Claims c = Verify(token);
        for (const auto &s : c.scopes)
            if (s == scope || s == "admin") return;
        throw Error(403, "token lacks required scope '" + scope + "'.");
    }

    std::string Auth::Identity(const std::string &token) const
    {
        if (!Enabled()) return "";
        return Verify(token).subject;
    }
}
