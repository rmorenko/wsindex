import logging
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration

logging.basicConfig(level=logging.INFO)

sentry_sdk.init(
    dsn="YOUR_SENTRY_PROJECT_DSN",
    integrations=[
        LoggingIntegration(
            level=logging.INFO,         # INFO+ become breadcrumbs
            event_level=logging.ERROR,  # ERROR+ generate events
        )
    ],
)

logger = logging.getLogger(__name__)

logger.info("Starting invoice export")
logger.warning("Customer 42 has no billing address")

try:
    export_invoices()
except Exception:
    logger.exception("Invoice export failed")