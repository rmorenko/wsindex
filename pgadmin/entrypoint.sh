#!/bin/sh
# Pre-register the Tensorus Postgres server in pgAdmin.
# servers.json is only imported on first login of the user, and PassFile
# paths are resolved relative to that user's storage directory.
set -e

STORAGE_DIR="/var/lib/pgadmin/storage/$(echo "$PGADMIN_DEFAULT_EMAIL" | tr '@' '_')"
mkdir -p "$STORAGE_DIR"
echo "db:5432:*:${POSTGRES_USER}:${POSTGRES_PASSWORD}" > "$STORAGE_DIR/pgpass"
chmod 600 "$STORAGE_DIR/pgpass"

cat > /var/lib/pgadmin/servers.json <<EOF
{
  "Servers": {
    "1": {
      "Name": "Tensorus",
      "Group": "Servers",
      "Host": "db",
      "Port": 5432,
      "MaintenanceDB": "${POSTGRES_DB}",
      "Username": "${POSTGRES_USER}",
      "SSLMode": "prefer",
      "PassFile": "/pgpass"
    }
  }
}
EOF

exec /entrypoint.sh "$@"
