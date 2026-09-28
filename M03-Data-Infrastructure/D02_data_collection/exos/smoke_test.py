from datetime import datetime


def main() -> None:
    print(f"Smoke test OK, time={datetime.utcnow().isoformat()}Z")


if __name__ == "__main__":
    main()

