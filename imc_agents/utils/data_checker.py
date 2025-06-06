import pandas as pd
import re
from imc_norm.product_number_check_service_impl import ProductNumberCheckServiceImpl

class DataChecker:

    """
    Diese Klasse überprüft CSV-Dateien mit Distributoren- und Kundeninformationen
    auf Vollständigkeit, Plausibilität und korrekte Formate.
    Sie nutzt u.a. externe API-Checks (MLFB).
    """

    def __init__(self, api_url: str, client_id: str, client_secret: str):
        self.product_service = ProductNumberCheckServiceImpl(api_url, client_id, client_secret)

    def read_csv_file(self, file_path: str) -> pd.DataFrame:

        """
        Liest die CSV-Datei ein (erst UTF-8, fallback ISO-8859-1),
        entfernt leere Spalten/Zeilen und prüft auf bekannte Fehlermeldungen.
        """

        try:
            with open(file_path, "rb") as f:
                start_bytes = f.read(1024)
                if b"Wrong File Format" in start_bytes:
                    raise ValueError("Datei enthält Fehlermeldung: 'Wrong File Format (SFTP only)'")

            try:
                df = pd.read_csv(file_path, delimiter=";", encoding="utf-8-sig", dtype=str)
            except UnicodeDecodeError:
                print("[DEBUG] utf-8-sig fehlgeschlagen, versuche ISO-8859-1")
                df = pd.read_csv(file_path, delimiter=";", encoding="ISO-8859-1", dtype=str)

            df.columns = [col.upper().strip() for col in df.columns]
            df = df.loc[:, ~df.columns.str.startswith('UNNAMED')]

            df = df.dropna(how='all')
            df = df[~(df.astype(str).apply(lambda x: x.str.strip() == '').all(axis=1))]

            print(f"[DEBUG] DataFrame geladen, Zeilen: {len(df)} Spalten: {len(df.columns)}")
            return df

        except Exception as e:
            print(f"[ERROR] Fehler beim Einlesen der Datei: {e}")
            # Signalisiere Fehler weiter
            raise RuntimeError(f"Dateifehler: {e}")

    def check_general_data(self, df: pd.DataFrame) -> dict:

        """
        Prüft allgemeine Pflichtspalten und das Format von Rechnungsnummern.
        """

        results = {}

        # Mandatory-Spalten
        mandatory_columns = [
            "DISTRIBUTOR_SENDER_ID", "DISTRIBUTOR_ORDER_TAKING_BRANCH_NAME", "DISTRIBUTOR_ORDER_TAKING_BRANCH_ID",
            "DISTRIBUTOR_SHIP_DATE", "DISTRIBUTOR_INVOICE_DATE", "DISTRIBUTOR_INVOICE_NUMBER",
            "DISTRIBUTOR_INVOICE_LINE_ITEM",
            "BILL_TO_CUSTOMER_DUNS_NUMBER", "BILL_TO_CUSTOMER_NATIONAL_REG_NUMBER", "BILL_TO_CUSTOMER_NAME",
            "BILL_TO_CUSTOMER_BILLING_ADDRESS1", "BILL_TO_CUSTOMER_BILLING_ADDRESS2", "BILL_TO_CUSTOMER_CITY",
            "BILL_TO_CUSTOMER_STATE", "BILL_TO_CUSTOMER_ZIP", "BILL_TO_CUSTOMER_COUNTRY",
            "SHIP_TO_CUSTOMER_DUNS_NUMBER", "SHIP_TO_CUSTOMER_NATIONAL_REG_NUMBER", "SHIP_TO_CUSTOMER_CUSTOMER_NAME",
            "SHIP_TO_CUSTOMER_ADDRESS1", "SHIP_TO_CUSTOMER_ADDRESS2", "SHIP_TO_CUSTOMER_CITY",
            "SHIP_TO_CUSTOMER_STATE", "SHIP_TO_CUSTOMER_ZIP", "SHIP_TO_CUSTOMER_COUNTRY",
            "VENDOR_ITEM_NUMBER", "VENDOR_ITEM_OPTIONS", "ITEM_DESCRIPTION", "PRODUCT_FAMILY",
            "QUANTITY", "QUANTITY_UNIT_OF_MEASURE", "UNIT_REPLENISHMENT_COST", "EXTENDED_REPLENISHMENT_COST",
            "UNIT_COST", "EXTENDED_COST_OF_GOODS_SOLD", "COST_UNIT_OF_MEASURE", "CURRENCY_CODE", "REBATE_NUMBER"
        ]

        # Fehlende Spalten prüfen
        missing_columns = [col for col in mandatory_columns if col not in df.columns]
        if missing_columns:
            results["missing_columns"] = missing_columns

        # Leere Werte in vorhandenen Pflichtspalten
        for col in mandatory_columns:
            if col in df.columns:
                missing_rows = df[
                    df[col].isnull() | (df[col].astype(str).str.strip() == "")
                    ]
                if not missing_rows.empty:
                    if len(missing_rows) == len(df):
                        # Alle Zeilen betroffen
                        results[f"missing_values_in_{col}"] = ["Alle Zeilen betroffen (100%)"]
                    else:
                        results[f"missing_values_in_{col}"] = [f"{len(missing_rows)} Zeilen betroffen"]
            else:
                results[f"missing_values_in_{col}"] = ["Spalte fehlt komplett"]

        # === Invoice Number Pattern Check ===
        numeric_pattern = re.compile(r"^\d+$")
        hex_pattern = re.compile(r"^[a-f0-9]{32}$", re.IGNORECASE)
        prefix_pattern = re.compile(r"^[A-Z]\d+$")

        if "DISTRIBUTOR_INVOICE_NUMBER" in df.columns:
            invoice_series = df["DISTRIBUTOR_INVOICE_NUMBER"].astype(str).str.strip()
            invalid_invoices = df[
                ~(
                        invoice_series.str.match(numeric_pattern, na=False) |
                        invoice_series.str.match(hex_pattern, na=False) |
                        invoice_series.str.match(prefix_pattern, na=False)
                )
            ]
            if not invalid_invoices.empty:
                results["invalid_invoice_numbers"] = [
                    f"Zeile {idx + 1}: {val}" for idx, val in
                    zip(invalid_invoices.index, invalid_invoices["DISTRIBUTOR_INVOICE_NUMBER"])
                ]

        return results

    def check_mlfb_numbers(self, df: pd.DataFrame) -> list:

        """
         Führt API-Check auf alle VENDOR_ITEM_NUMBER-Einträge durch
         und gibt Liste ungültiger MLFB-Nummern zurück.
         """

        if "VENDOR_ITEM_NUMBER" not in df.columns:
            return ["Fehler: Spalte 'VENDOR_ITEM_NUMBER' nicht gefunden"]

        item_series = df["VENDOR_ITEM_NUMBER"].dropna().astype(str).str.strip()
        all_numbers = item_series.tolist()
        all_indices = item_series.index.tolist()

        if not all_numbers:
            return ["Keine VENDOR_ITEM_NUMBER vorhanden, MLFB-Prüfung übersprungen"]

        api_results = self.product_service.validate_product_numbers_batch(all_numbers)

        invalid_entries = []
        valid_systems = ["MLFB", "TNS", "SFC", "SSN"]
        for idx, num, result in zip(all_indices, all_numbers, api_results):
            output = result.get("output", {})
            system = output.get("system")
            if system not in valid_systems :
                invalid_entries.append(f"Zeile {idx + 1}: {num}")

        if invalid_entries:
            return [f"Ungültige MLFB-Nummern:\n" + "\n".join(invalid_entries)]
        else:
            return ["Alle MLFB-Nummern sind gültig"]

    def check_distributor_data(self, df: pd.DataFrame) -> dict:

        """
        Prüft Distributoren-bezogene Spalten (z. B. Invoice-Datum, Sender-ID)
        auf Vollständigkeit und korrektes Format.
        """

        results = {}

        if "DISTRIBUTOR_INVOICE_DATE" in df.columns:
            date_pattern = re.compile(r"\d{4}-\d{2}-\d{2}")
            invalid_dates = df[
                ~df["DISTRIBUTOR_INVOICE_DATE"].astype(str).str.match(date_pattern, na=False)
            ]
            results["invalid_distributor_invoice_dates"] = [f"Zeile {idx + 1}: {val}" for idx, val in zip(invalid_dates.index, invalid_dates["DISTRIBUTOR_INVOICE_DATE"])]
        else:
            results["invalid_distributor_invoice_dates"] = ["Spalte 'DISTRIBUTOR_INVOICE_DATE' fehlt"]

        if "DISTRIBUTOR_SENDER_ID" in df.columns:
            missing_sender_id = df[
                df["DISTRIBUTOR_SENDER_ID"].isnull() | (df["DISTRIBUTOR_SENDER_ID"].astype(str).str.strip() == "")
            ]
            results["missing_sender_id"] = [f"Zeile {idx + 1}" for idx in missing_sender_id.index]
        else:
            results["missing_sender_id"] = ["Spalte 'DISTRIBUTOR_SENDER_ID' fehlt"]

        if "DISTRIBUTOR_INVOICE_NUMBER" in df.columns:
            missing_invoice_number = df[
                df["DISTRIBUTOR_INVOICE_NUMBER"].isnull() | (df["DISTRIBUTOR_INVOICE_NUMBER"].astype(str).str.strip() == "")
            ]
            results["missing_invoice_number"] = [f"Zeile {idx + 1}" for idx in missing_invoice_number.index]
        else:
            results["missing_invoice_number"] = ["Spalte 'DISTRIBUTOR_INVOICE_NUMBER' fehlt"]

        return results

    def check_customer_data(self, df: pd.DataFrame) -> dict:

        """
        Prüft Kundendaten: Länder, Staaten (US/MX), Namen, ZIPs,
        ob sie gültig und vollständig sind.
        """

        results = {}
        known_countries = {"DE", "US", "MX", "GB", "UK", "FR", "IT", "ES", "CN", "JP"}

        # BILL_TO checks
        if "BILL_TO_CUSTOMER_COUNTRY" in df.columns:
            invalid_countries = df[
                ~df["BILL_TO_CUSTOMER_COUNTRY"].astype(str).str.upper().isin(known_countries)
            ]
            results["invalid_bill_to_countries"] = [f"Zeile {idx + 1}: {val}" for idx, val in
                                                    zip(invalid_countries.index,
                                                        invalid_countries["BILL_TO_CUSTOMER_COUNTRY"])]
        else:
            results["invalid_bill_to_countries"] = ["Spalte 'BILL_TO_CUSTOMER_COUNTRY' fehlt"]

        if "BILL_TO_CUSTOMER_NAME" in df.columns:
            missing_names = df[
                df["BILL_TO_CUSTOMER_NAME"].isnull() | (df["BILL_TO_CUSTOMER_NAME"].astype(str).str.strip() == "")
                ]
            results["missing_bill_to_names"] = [f"Zeile {idx + 1}" for idx in missing_names.index]
        else:
            results["missing_bill_to_names"] = ["Spalte 'BILL_TO_CUSTOMER_NAME' fehlt"]

        if "BILL_TO_CUSTOMER_ZIP" in df.columns:
            missing_zip = df[
                df["BILL_TO_CUSTOMER_ZIP"].isnull() | (df["BILL_TO_CUSTOMER_ZIP"].astype(str).str.strip() == "")
                ]
            results["missing_bill_to_zip"] = [f"Zeile {idx + 1}" for idx in missing_zip.index]
        else:
            results["missing_bill_to_zip"] = ["Spalte 'BILL_TO_CUSTOMER_ZIP' fehlt"]

        if "SHIP_TO_CUSTOMER_COUNTRY" in df.columns:
            invalid_countries = df[
                ~df["SHIP_TO_CUSTOMER_COUNTRY"].astype(str).str.upper().isin(known_countries)
            ]
            results["invalid_ship_to_countries"] = [f"Zeile {idx + 1}: {val}" for idx, val in
                                                    zip(invalid_countries.index,
                                                        invalid_countries["SHIP_TO_CUSTOMER_COUNTRY"])]
        else:
            results["invalid_ship_to_countries"] = ["Spalte 'SHIP_TO_CUSTOMER_COUNTRY' fehlt"]

        if "SHIP_TO_CUSTOMER_CUSTOMER_NAME" in df.columns:
            missing_ship_names = df[
                df["SHIP_TO_CUSTOMER_CUSTOMER_NAME"].isnull() | (
                            df["SHIP_TO_CUSTOMER_CUSTOMER_NAME"].astype(str).str.strip() == "")
                ]
            results["missing_ship_to_names"] = [f"Zeile {idx + 1}" for idx in missing_ship_names.index]
        else:
            results["missing_ship_to_names"] = ["Spalte 'SHIP_TO_CUSTOMER_CUSTOMER_NAME' fehlt"]

        if "SHIP_TO_CUSTOMER_ZIP" in df.columns:
            missing_zip = df[
                df["SHIP_TO_CUSTOMER_ZIP"].isnull() | (df["SHIP_TO_CUSTOMER_ZIP"].astype(str).str.strip() == "")
                ]
            results["missing_ship_to_zip"] = [f"Zeile {idx + 1}" for idx in missing_zip.index]
        else:
            results["missing_ship_to_zip"] = ["Spalte 'SHIP_TO_CUSTOMER_ZIP' fehlt"]

        valid_us_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
        }

        valid_mexican_states = {
            'AGUASCALIENTES', 'BAJA CALIFORNIA', 'BAJA CALIFORNIA SUR', 'CAMPECHE',
            'CHIAPAS', 'CHIHUAHUA', 'COAHUILA', 'COLIMA', 'DURANGO', 'GUANAJUATO',
            'GUERRERO', 'HIDALGO', 'JALISCO', 'MEXICO', 'MICHOACAN', 'MORELOS',
            'NAYARIT', 'NUEVO LEON', 'OAXACA', 'PUEBLA', 'QUERETARO', 'QUINTANA ROO',
            'SAN LUIS POTOSI', 'SINALOA', 'SONORA', 'TABASCO', 'TAMAULIPAS',
            'TLAXCALA', 'VERACRUZ', 'YUCATAN', 'ZACATECAS'
        }

        valid_states = valid_us_states.union(valid_mexican_states)

        # === SHIP_TO ===
        if "SHIP_TO_CUSTOMER_STATE" in df.columns:
            state_series = df["SHIP_TO_CUSTOMER_STATE"].astype(str).str.strip().str.upper()

            missing_state = df[state_series == ""]
            if not missing_state.empty:
                if len(missing_state) == len(df):
                    results["missing_ship_to_state"] = ["Alle Zeilen betroffen (100%)"]
                else:
                    results["missing_ship_to_state"] = [f"{len(missing_state)} Zeilen betroffen"]

            invalid_mask = ~state_series.isin(valid_states) & (state_series != "")
            results["invalid_ship_to_state"] = self.summarize_invalid_entries(df, "SHIP_TO_CUSTOMER_STATE",
                                                                              invalid_mask)

        # === BILL_TO ===
        if "BILL_TO_CUSTOMER_STATE" in df.columns:
            state_series = df["BILL_TO_CUSTOMER_STATE"].astype(str).str.strip().str.upper()

            missing_state = df[state_series == ""]
            if not missing_state.empty:
                if len(missing_state) == len(df):
                    results["missing_bill_to_state"] = ["Alle Zeilen betroffen (100%)"]
                else:
                    results["missing_bill_to_state"] = [f"{len(missing_state)} Zeilen betroffen"]

            invalid_mask = ~state_series.isin(valid_states) & (state_series != "")
            results["invalid_bill_to_state"] = self.summarize_invalid_entries(df, "BILL_TO_CUSTOMER_STATE",
                                                                              invalid_mask)

        return results

    def check_financial_data(self, df: pd.DataFrame) -> dict:

        """
        Prüft Finanzdaten wie Währungen, Kostenfelder und Mengen
        auf numerische Korrektheit und gültige Codes.
        """

        results = {}
        known_currencies = {"USD", "EUR", "GBP", "JPY", "MXN", "CAD", "AUD", "CHF"}

        if "CURRENCY_CODE" in df.columns:
            invalid_currencies = df[
                ~df["CURRENCY_CODE"].astype(str).str.upper().isin(known_currencies)
            ]
            results["invalid_currencies"] = [f"Zeile {idx + 1}: {val}" for idx, val in zip(invalid_currencies.index, invalid_currencies["CURRENCY_CODE"])]
        else:
            results["invalid_currencies"] = ["Spalte 'CURRENCY_CODE' fehlt"]

        for col in ["UNIT_COST", "EXTENDED_COST_OF_GOODS_SOLD"]:
            if col in df.columns:
                invalid_values = df[
                    pd.to_numeric(df[col], errors="coerce").isnull()
                ]
                results[f"invalid_{col.lower()}"] = [f"Zeile {idx + 1}: {val}" for idx, val in zip(invalid_values.index, invalid_values[col])]
            else:
                results[f"invalid_{col.lower()}"] = [f"Spalte '{col}' fehlt"]

        if "QUANTITY" in df.columns:
            invalid_quantities = df[
                pd.to_numeric(df["QUANTITY"], downcast="integer", errors="coerce").isnull()
            ]
            results["invalid_quantity"] = [f"Zeile {idx + 1}: {val}" for idx, val in zip(invalid_quantities.index, invalid_quantities["QUANTITY"])]
        else:
            results["invalid_quantity"] = ["Spalte 'QUANTITY' fehlt"]

        return results

    def summarize_invalid_entries(self, df, column, invalid_mask):
        invalid_entries = df[invalid_mask]
        unique_values = invalid_entries[column].unique()

        if len(invalid_entries) == len(df):
            return ["Alle Zeilen betroffen (100%)"]
        elif len(unique_values) == 1:
            return [f"Alle Zeilen enthalten nur '{unique_values[0]}' als ungültigen Wert"]
        else:
            return [f"Zeile {idx + 1}: {val}" for idx, val in zip(invalid_entries.index, invalid_entries[column])]

    
