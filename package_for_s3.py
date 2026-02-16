#!/usr/bin/env python3
"""Package src/ and config/ into extra_libs.zip for AWS Glue Python Shell.

Usage:
    python package_for_glue.py [output_dir]
Output:
    extra_libs.zip in current dir or output_dir (Python 3.9+).
"""

import zipfile
import sys
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir
    zip_path = out_dir / "extra_libs.zip"

    src_dir = script_dir / "src"
    config_dir = script_dir / "config"
    schema_postgres = script_dir / "schema_postgres.sql"
    schema_sqlite = script_dir / "schema.sql"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in (src_dir, config_dir):
            if not d.exists():
                continue
            for f in d.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(script_dir))
        for schema in (schema_postgres, schema_sqlite):
            if schema.exists():
                zf.write(schema, schema.name)

    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()
