"""Production Kaggle dataset ingestion & idempotent database seeder for ecom.

Reads and seeds ALL records from 'Kaggle dataset archive/':
1. DIM_LOCATION.csv -> locations (113 rows)
2. DIM_PAYMENT.csv -> payment_methods (14 rows)
3. DIM_FULFILLMENT.csv -> fulfillment_methods (3 rows)
4. DIM_SELLER.csv -> sellers (4,563 rows)
5. DIM_PRODUCT.csv -> products (1,260 items)
6. DIM_CUSTOMER.csv -> customers (ALL 90,832 records)
7. users -> credentials table (admins + customer logins)
8. FACT_ORDERS.csv -> orders (ALL 342,033 orders) & order_items (ALL 790,732 lines)
9. FACT_RETURNS.csv -> returns & refunds (ALL 103,035 records)
10. data/policies/ -> store policies into pgvector

High-performance bulk ingestion using native PostgreSQL COPY protocol via asyncpg.
"""

import asyncio
from datetime import datetime, timezone
import csv
import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import asyncpg
from sqlalchemy import func, select

from app.auth import hash_password
from app.config import settings
from app.db.models import User
from app.db.session import AsyncSessionLocal, init_db, reset_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ecom.seed")

# Resolve Kaggle dataset archive path
ECOM_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = ECOM_DIR.parent / "Kaggle dataset archive"
if not DATASET_DIR.exists():
    DATASET_DIR = ECOM_DIR / "Kaggle dataset archive"


def parse_date(val: Optional[str]) -> Optional[datetime]:
    """Parse various Kaggle date string formats into timezone-aware UTC datetime."""
    if not val or not val.strip():
        return None
    val = val.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%Y%m%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def derive_warranty_months(category: str, sub_cat: str) -> int:
    """Derive warranty duration by product category."""
    cat = (category or "").lower()
    sub = (sub_cat or "").lower()
    if "electronic" in cat or "mobile" in sub or "gaming" in cat or "laptop" in sub:
        return 12
    if "appliance" in cat or "automotive" in cat:
        return 24
    if "cookware" in sub or "furniture" in sub:
        return 36
    if "clothing" in cat or "footwear" in cat:
        return 6
    return 0


async def seed_database(reset: bool = False, force: bool = False) -> Dict[str, int]:
    """Idempotently seed the full Kaggle dataset into PostgreSQL using COPY bulk loading."""
    start_time = time.perf_counter()
    report: Dict[str, int] = {}
    should_reset = reset or force

    if should_reset:
        logger.info("Executing requested full database reset (dropping & re-creating all tables)...")
        await reset_db()
    else:
        await init_db()

    # Check if already seeded with full dataset
    async with AsyncSessionLocal() as session:
        user_count = await session.scalar(select(func.count()).select_from(User))
        if user_count and user_count > 0 and not should_reset:
            logger.info("Database already contains %d users. Skipping full seed (use --reset to force).", user_count)
            return {"users": user_count, "status": "already_seeded"}

    # Connect directly with asyncpg for high-speed COPY streaming
    raw_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    conn: asyncpg.Connection = await asyncpg.connect(raw_url)

    try:
        logger.info("Commencing Full Kaggle Dataset Ingestion from: %s", DATASET_DIR)

        # ── 1. DIM_LOCATION.csv ───────────────────────────────────────────────
        t0 = time.perf_counter()
        loc_path = DATASET_DIR / "DIM_LOCATION.csv"
        location_ids: Set[str] = set()
        loc_records = []
        if loc_path.exists():
            with open(loc_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    lid = row.get("location_id", "").strip()
                    if not lid:
                        continue
                    location_ids.add(lid)
                    loc_records.append((
                        lid,
                        row.get("state") or None,
                        row.get("city") or None,
                        row.get("postal_code") or None,
                        row.get("region") or None,
                        float(row["latitude"]) if row.get("latitude") else None,
                        float(row["longitude"]) if row.get("longitude") else None,
                        row.get("area_type") or None,
                        row.get("location_category") or None,
                    ))
            await conn.copy_records_to_table(
                "locations",
                records=loc_records,
                columns=["id", "state", "city", "postal_code", "region", "latitude", "longitude", "area_type", "location_category"]
            )
            report["locations"] = len(loc_records)
            logger.info("Seeded %d locations in %.2fs", len(loc_records), time.perf_counter() - t0)

        # ── 2. DIM_PAYMENT.csv ────────────────────────────────────────────────
        t0 = time.perf_counter()
        pay_path = DATASET_DIR / "DIM_PAYMENT.csv"
        payment_ids: Set[str] = set()
        pay_records = []
        if pay_path.exists():
            with open(pay_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid = row.get("payment_id", "").strip()
                    if not pid:
                        continue
                    payment_ids.add(pid)
                    pay_records.append((
                        pid,
                        row.get("payment_method", "Credit Card"),
                        row.get("payment_provider") or None,
                        row.get("description") or None,
                    ))
            await conn.copy_records_to_table(
                "payment_methods",
                records=pay_records,
                columns=["id", "payment_method", "payment_provider", "description"]
            )
            report["payment_methods"] = len(pay_records)
            logger.info("Seeded %d payment methods in %.2fs", len(pay_records), time.perf_counter() - t0)

        # ── 3. DIM_FULFILLMENT.csv ────────────────────────────────────────────
        t0 = time.perf_counter()
        ful_path = DATASET_DIR / "DIM_FULFILLMENT.csv"
        fulfillment_ids: Set[str] = set()
        ful_records = []
        if ful_path.exists():
            with open(ful_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    fid = row.get("fulfillment_id", "").strip()
                    if not fid:
                        continue
                    fulfillment_ids.add(fid)
                    ful_records.append((
                        fid,
                        row.get("shipping_method", "Standard"),
                        row.get("service_level") or None,
                        int(row["delivery_sla_days"]) if row.get("delivery_sla_days") else 3,
                        float(row["base_shipping_cost"]) if row.get("base_shipping_cost") else 0.0,
                    ))
            await conn.copy_records_to_table(
                "fulfillment_methods",
                records=ful_records,
                columns=["id", "shipping_method", "service_level", "delivery_sla_days", "base_shipping_cost"]
            )
            report["fulfillment_methods"] = len(ful_records)
            logger.info("Seeded %d fulfillment methods in %.2fs", len(ful_records), time.perf_counter() - t0)

        # ── 4. DIM_SELLER.csv ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        sel_path = DATASET_DIR / "DIM_SELLER.csv"
        sel_records = []
        if sel_path.exists():
            with open(sel_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sid = row.get("seller_id", "").strip()
                    if not sid:
                        continue
                    lid = row.get("location_id")
                    sel_records.append((
                        sid,
                        row.get("seller_name", f"Seller {sid}"),
                        row.get("email") or None,
                        row.get("phone_number") or None,
                        row.get("join_date") or None,
                        lid if lid in location_ids else None,
                        float(row["rating"]) if row.get("rating") else 0.0,
                        row.get("category_focus") or None,
                    ))
            await conn.copy_records_to_table(
                "sellers",
                records=sel_records,
                columns=["id", "name", "email", "phone_number", "join_date", "location_id", "rating", "category_focus"]
            )
            report["sellers"] = len(sel_records)
            logger.info("Seeded %d sellers in %.2fs", len(sel_records), time.perf_counter() - t0)

        # ── 5. DIM_PRODUCT.csv ────────────────────────────────────────────────
        t0 = time.perf_counter()
        prod_path = DATASET_DIR / "DIM_PRODUCT.csv"
        product_names_map: Dict[str, str] = {}
        prod_records = []
        if prod_path.exists():
            with open(prod_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    pid = row.get("product_id", "").strip()
                    if not pid:
                        continue
                    pname = row.get("product_name", f"Product {pid}")
                    brand = row.get("brand", "Generic")
                    category = row.get("category", "General")
                    sub_category = row.get("sub_category", "General")
                    price = float(row.get("list_price") or 49.99)
                    warranty = derive_warranty_months(category, sub_category)
                    desc = f"{brand} {pname} - Premium {category} ({sub_category}). Standard {warranty}-month warranty."
                    product_names_map[pid] = pname

                    prod_records.append((
                        pid, pname, brand, category, sub_category, price, 50, warranty, desc
                    ))
            await conn.copy_records_to_table(
                "products",
                records=prod_records,
                columns=["id", "name", "brand", "category", "sub_category", "price", "stock_quantity", "warranty_months", "description"]
            )
            report["products"] = len(prod_records)
            logger.info("Seeded %d products in %.2fs", len(prod_records), time.perf_counter() - t0)

        # ── 6. DIM_CUSTOMER.csv (ALL 90,832 records) ──────────────────────────
        t0 = time.perf_counter()
        cust_path = DATASET_DIR / "DIM_CUSTOMER.csv"
        customer_email_map: Dict[str, str] = {}
        cust_records = []
        sample_for_credentials = []

        if cust_path.exists():
            with open(cust_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    cid = row.get("customer_id", "").strip()
                    email = (row.get("email") or "").strip().lower()
                    if not cid or not email:
                        continue

                    fname = row.get("first_name", "Customer")
                    lname = row.get("last_name", cid)
                    lid = row.get("location_id")
                    customer_email_map[cid] = email

                    if len(sample_for_credentials) < 25:
                        sample_for_credentials.append((cid, f"{fname} {lname}".strip(), email))

                    cust_records.append((
                        cid,
                        fname,
                        lname,
                        email,
                        row.get("phone_number") or None,
                        row.get("gender") or None,
                        row.get("date_of_birth") or None,
                        row.get("registration_date") or None,
                        row.get("income_bracket") or None,
                        row.get("marital_status") or None,
                        lid if lid in location_ids else None,
                        row.get("upi_id") or None,
                        row.get("credit_card_number") or None,
                    ))

        # Add demo test customers Alice and Bob
        cust_records.append((
            "CUST-ALICE-01", "Alice", "Johnson", "alice@email.com", "+1 555 0192",
            "F", "1990-01-01", "2024-01-01", "Middle", "Single", None, "alice@upi", "4111-2222-3333-4444"
        ))
        customer_email_map["CUST-ALICE-01"] = "alice@email.com"

        cust_records.append((
            "CUST-BOB-02", "Bob", "Martinez", "bob@email.com", "+1 555 0193",
            "M", "1988-05-14", "2024-01-01", "Upper-Middle", "Married", None, "bob@upi", "5111-2222-3333-5555"
        ))
        customer_email_map["CUST-BOB-02"] = "bob@email.com"

        await conn.copy_records_to_table(
            "customers",
            records=cust_records,
            columns=[
                "id", "first_name", "last_name", "email", "phone_number", "gender",
                "date_of_birth", "registration_date", "income_bracket", "marital_status",
                "location_id", "upi_id", "credit_card_number"
            ]
        )
        report["customers"] = len(cust_records)
        logger.info("Seeded ALL %d customers in %.2fs", len(cust_records), time.perf_counter() - t0)

        # ── 7. Users / Credentials Table ──────────────────────────────────────
        t0 = time.perf_counter()
        now_utc = datetime.now(timezone.utc)
        user_records = [
            ("USR-ADMIN-01", None, "Admin One", "admin1@ecom.com", hash_password("admin123"), "admin", now_utc),
            ("USR-ADMIN-02", None, "Admin Two", "admin2@ecom.com", hash_password("admin123"), "admin", now_utc),
            ("USR-ALICE-01", "CUST-ALICE-01", "Alice Johnson", "alice@email.com", hash_password("user123"), "customer", now_utc),
            ("USR-BOB-02", "CUST-BOB-02", "Bob Martinez", "bob@email.com", hash_password("user123"), "customer", now_utc),
        ]
        for cid, name, email in sample_for_credentials:
            user_records.append((
                f"USR-{cid}", cid, name or email.split("@")[0], email, hash_password("user123"), "customer", now_utc
            ))

        await conn.copy_records_to_table(
            "users",
            records=user_records,
            columns=["id", "customer_id", "name", "email", "password_hash", "role", "created_at"]
        )
        report["users"] = len(user_records)
        logger.info("Seeded %d user credentials in %.2fs", len(user_records), time.perf_counter() - t0)

        # ── 8. FACT_ORDERS.csv (ALL 342,033 orders & 790,732 lines) ───────────
        t0 = time.perf_counter()
        orders_path = DATASET_DIR / "FACT_ORDERS.csv"
        orders_dict: Dict[str, list] = {}
        order_items_list: List[tuple] = []

        if orders_path.exists():
            logger.info("Streaming and parsing FACT_ORDERS.csv (790,000+ records)...")
            with open(orders_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    oid = row["order_id"]
                    cid = row["customer_id"]
                    pid = row["product_id"]
                    line_id = row["order_line_id"]
                    net_amt = float(row["net_amount"] or 0.0)
                    cust_email = customer_email_map.get(cid, f"{cid.lower()}@ecom.com")
                    prod_name = product_names_map.get(pid, f"Product {pid}")

                    if oid not in orders_dict:
                        o_date = parse_date(row["order_date"]) or now_utc
                        act_del = parse_date(row["actual_delivery_date"])
                        exp_del = parse_date(row["expected_delivery_date"])
                        r_status = (row["order_status"] or "Completed").strip().lower()
                        status = "delivered" if (r_status == "completed" and act_del) else ("completed" if r_status == "completed" else r_status)

                        orders_dict[oid] = [
                            oid,
                            cid,
                            cust_email,
                            o_date,
                            status,
                            net_amt,
                            float(row["shipping_fee"] or 0.0),
                            float(row["tax_amount"] or 0.0),
                            float(row["discount_amount"] or 0.0),
                            f"TRK-{oid[3:]}",
                            exp_del,
                            act_del,
                            int(row["delivery_sla_days"] or 2),
                            int(row["delivery_delay_days"] or 0),
                            row["payment_id"] if row.get("payment_id") in payment_ids else None,
                            row["fulfillment_id"] if row.get("fulfillment_id") in fulfillment_ids else None,
                            o_date,
                        ]
                    else:
                        orders_dict[oid][5] = round(orders_dict[oid][5] + net_amt, 2)
                        orders_dict[oid][7] = round(orders_dict[oid][7] + float(row["tax_amount"] or 0.0), 2)
                        orders_dict[oid][8] = round(orders_dict[oid][8] + float(row["discount_amount"] or 0.0), 2)

                    order_items_list.append((
                        line_id,
                        oid,
                        pid,
                        prod_name,
                        int(row["quantity"] or 1),
                        float(row["unit_price"] or 0.0),
                        float(row["discount_percentage"] or 0.0),
                        float(row["tax_percentage"] or 0.0),
                        net_amt,
                    ))

        # Alice & Bob test fixture orders
        first_pid = next(iter(product_names_map))
        second_pid = list(product_names_map.keys())[1]

        orders_dict["ORD-2024-001"] = [
            "ORD-2024-001", "CUST-ALICE-01", "alice@email.com", now_utc, "delivered",
            79.99, 5.99, 4.0, 0.0, "TRK-98214001", now_utc, now_utc, 2, 0, "PAY001", "FUL002", now_utc
        ]
        order_items_list.append((
            "OL-ALICE-01", "ORD-2024-001", first_pid, product_names_map[first_pid],
            1, 79.99, 0.0, 5.0, 79.99
        ))

        orders_dict["ORD-2024-002"] = [
            "ORD-2024-002", "CUST-BOB-02", "bob@email.com", now_utc, "delivered",
            149.99, 8.99, 7.5, 0.0, "TRK-98214002", now_utc, now_utc, 2, 0, "PAY002", "FUL001", now_utc
        ]
        order_items_list.append((
            "OL-BOB-01", "ORD-2024-002", second_pid, product_names_map[second_pid],
            1, 149.99, 0.0, 5.0, 149.99
        ))

        # Bulk copy orders
        orders_tuples = [tuple(v) for v in orders_dict.values()]
        logger.info("Executing COPY for %d orders...", len(orders_tuples))
        await conn.copy_records_to_table(
            "orders",
            records=orders_tuples,
            columns=[
                "id", "customer_id", "customer_email", "order_date", "status", "total_amount",
                "shipping_fee", "tax_amount", "discount_amount", "tracking_number",
                "expected_delivery_date", "actual_delivery_date", "delivery_sla_days",
                "delivery_delay_days", "payment_id", "fulfillment_id", "created_at"
            ]
        )
        report["orders"] = len(orders_tuples)

        # Bulk copy order items
        logger.info("Executing COPY for %d order items...", len(order_items_list))
        await conn.copy_records_to_table(
            "order_items",
            records=order_items_list,
            columns=[
                "id", "order_id", "product_id", "product_name", "quantity",
                "unit_price", "discount_percentage", "tax_percentage", "net_amount"
            ]
        )
        report["order_items"] = len(order_items_list)
        logger.info("Seeded ALL %d orders with %d order items in %.2fs", len(orders_tuples), len(order_items_list), time.perf_counter() - t0)

        # ── 9. FACT_RETURNS.csv (ALL 103,035 records) ─────────────────────────
        t0 = time.perf_counter()
        returns_path = DATASET_DIR / "FACT_RETURNS.csv"
        returns_list: List[tuple] = []
        refunds_list: List[tuple] = []

        if returns_path.exists():
            logger.info("Streaming and parsing FACT_RETURNS.csv (103,000+ records)...")
            with open(returns_path, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    line_id = row["order_line_id"]
                    oid = row["order_id"]
                    cid = row["customer_id"]
                    pid = row["product_id"]
                    if oid not in orders_dict or cid not in customer_email_map:
                        continue

                    rma_code = f"RMA-{line_id.replace('OL', '')}"
                    ref_id = f"REF-{line_id.replace('OL', '')}"
                    reason = row.get("review_text") or "Customer return requested"
                    days_ret = int(row.get("days_to_return") or 2)
                    restock = float(row.get("restocking_fee") or 0.0)

                    raw_amt = float(row.get("refund_amount") or 0.0)
                    amt = abs(raw_amt) if raw_amt != 0 else float(row.get("unit_price") or 50.0)
                    ref_status = "completed" if amt <= 50.0 else "pending_admin_approval"
                    dt = parse_date(row.get("date_id")) or now_utc

                    returns_list.append((
                        rma_code, oid, line_id, cid, pid if pid in product_names_map else None,
                        reason, "approved", days_ret, restock, dt
                    ))
                    refunds_list.append((
                        ref_id, rma_code, oid, round(amt, 2), ref_status, f"seed-key-{line_id}", dt
                    ))

        # Admin test fixture return and refund
        returns_list.append((
            "RMA-ADMIN-TEST", "ORD-2024-001", "OL-ALICE-01", "CUST-ALICE-01", first_pid,
            "Defective component on delivery", "requested", 2, 0.0, now_utc
        ))
        refunds_list.append((
            "REF-ADMIN-TEST", "RMA-ADMIN-TEST", "ORD-2024-001", 149.99,
            "pending_admin_approval", "seed-key-admin-test-01", now_utc
        ))

        logger.info("Executing COPY for %d returns and %d refunds...", len(returns_list), len(refunds_list))
        await conn.copy_records_to_table(
            "returns",
            records=returns_list,
            columns=["id", "order_id", "order_line_id", "customer_id", "product_id", "reason", "status", "days_to_return", "restocking_fee", "created_at"]
        )
        await conn.copy_records_to_table(
            "refunds",
            records=refunds_list,
            columns=["id", "return_id", "order_id", "amount", "status", "idempotency_key", "created_at"]
        )
        report["returns"] = len(returns_list)
        report["refunds"] = len(refunds_list)
        logger.info("Seeded ALL %d returns & %d refunds in %.2fs", len(returns_list), len(refunds_list), time.perf_counter() - t0)

    finally:
        await conn.close()

    # ── 10. Ingest Policy Documents ───────────────────────────────────────────
    async with AsyncSessionLocal() as session:
        logger.info("Ingesting store policy documents into vector storage...")
        from app.rag.ingest import ingest_policies
        policy_count = await ingest_policies(session)
        report["policies"] = policy_count

    elapsed = round(time.perf_counter() - start_time, 2)
    report["elapsed_seconds"] = elapsed
    logger.info("==================================================")
    logger.info("FULL DATASET SEEDING COMPLETED IN %.2f SECONDS", elapsed)
    logger.info("Seeding Report: %s", report)
    logger.info("==================================================")

    return report


if __name__ == "__main__":
    reset_flag = "--reset" in sys.argv
    asyncio.run(seed_database(reset=reset_flag))
