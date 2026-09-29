import json
import os
from decimal import Decimal

import boto3
import psycopg2
from dotenv import load_dotenv
import pandas as pd  # ← AJOUT 1
import pandera.pandas as pa  # ← AJOUT 1


load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX")

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# ← AJOUT 2 : le contrat de données
employee_schema = pa.DataFrameSchema(
    {
        "employee_id": pa.Column(str, pa.Check.str_startswith("EMP-")),
        "team": pa.Column(str),
        "seniority_level": pa.Column(str),
        "base_salary": pa.Column(float, pa.Check.ge(0), coerce=True),
        "bonus": pa.Column(float, pa.Check.ge(0), coerce=True),
        "status": pa.Column(str, pa.Check.isin(["active"])),
        "contract_type": pa.Column(str, pa.Check.isin(["PERMANENT", "CONTRACT"])),
    }
)


def extract_from_s3():
    s3 = boto3.client("s3")
    employees_raw = []

    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_PREFIX)

    for item in response.get("Contents", []):
        key = item["Key"]
        if not key.endswith(".json"):
            continue
        s3_object = s3.get_object(Bucket=S3_BUCKET, Key=key)
        body_text = s3_object["Body"].read().decode("utf-8")
        employees_raw.append(json.loads(body_text))

    return employees_raw


def transform(employees_raw):
    employees_clean = []

    for employee in employees_raw:
        status = employee["status"]
        if status != "active":
            continue

        contract_type = employee["contractType"]
        if contract_type == "INTERN":
            continue

        base_salary = Decimal(employee["baseSalary"].split(" ")[0])
        bonus = Decimal(employee["bonus"].split(" ")[0])

        employees_clean.append(
            {
                "employee_id": employee["employeeId"],
                "team": employee["team"],
                "seniority_level": employee["seniorityLevel"],
                "base_salary": base_salary,
                "bonus": bonus,
                "status": status,
                "contract_type": contract_type,
            }
        )

    return employees_clean


# ← AJOUT 3 : la quality gate, entre transform et load
def quality_gate(employees_clean):
    df = pd.DataFrame(employees_clean)

    try:
        validated_df = employee_schema.validate(df, lazy=True)
        return validated_df.to_dict("records")
    except pa.errors.SchemaErrors as err:
        bad_indexes = err.failure_cases["index"].dropna().astype(int).unique()

        os.makedirs("rejected", exist_ok=True)
        rejected_df = df.loc[bad_indexes]
        rejected_df.to_csv("rejected/employees_rejected.csv", index=False)
        print(f"Quality gate: {len(rejected_df)} row(s) quarantined in rejected/")

        clean_df = df.drop(index=bad_indexes)
        return clean_df.to_dict("records")


def load_to_postgres(employees_clean):
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    cur = conn.cursor()

    sql = """
    INSERT INTO employees_v2 (
        employee_id, team, seniority_level, base_salary, bonus, status, contract_type
    )
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (employee_id) DO UPDATE SET
        team = EXCLUDED.team,
        seniority_level = EXCLUDED.seniority_level,
        base_salary = EXCLUDED.base_salary,
        bonus = EXCLUDED.bonus,
        status = EXCLUDED.status,
        contract_type = EXCLUDED.contract_type;
    """

    for row in employees_clean:
        cur.execute(
            sql,
            (
                row["employee_id"],
                row["team"],
                row["seniority_level"],
                row["base_salary"],
                row["bonus"],
                row["status"],
                row["contract_type"],
            ),
        )

    conn.commit()
    cur.close()
    conn.close()


# ← AJOUT 4 : main passe par la quality gate
def main():
    print("Extracting data from S3...")
    employees_raw = extract_from_s3()
    print(f"Raw employees found: {len(employees_raw)}")

    print("Transforming data...")
    employees_clean = transform(employees_raw)
    print(f"Clean employees after transform: {len(employees_clean)}")

    print("Running the quality gate...")
    employees_validated = quality_gate(employees_clean)
    print(f"Validated employees ready to load: {len(employees_validated)}")

    print("Loading data into PostgreSQL...")
    load_to_postgres(employees_validated)

    print("ETL done.")


if __name__ == "__main__":
    main()
