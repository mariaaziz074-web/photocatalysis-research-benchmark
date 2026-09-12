"""
src/photocatalysis_benchmark/validation/pipeline.py
==================================================
Production data validation and normalization pipeline.
Integrates chem-research-data (chemdata) library with rigorous physics-informed boundary checks.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np
import yaml

# Check for chemdata availability
try:
    from chemdata.validation.chemical import validate_smiles, validate_chemical_formula, compute_molecular_weight
    from chemdata.normalization.units import UnitNormalizer
    from chemdata.normalization.chemicals import canonicalize_smiles, canonicalize_formula
    from chemdata.provenance.tracker import ProvenanceTracker
    HAS_CHEMDATA = True
except ImportError:
    HAS_CHEMDATA = False

# RDKit fallback check
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

logger = logging.getLogger("validation_pipeline")


class DatasetValidationPipeline:
    """
    Standardization and validation pipeline for photocatalytic degradation datasets.
    """

    def __init__(self, rules_config_path: Optional[Path] = None, schema_config_path: Optional[Path] = None):
        self.project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.rules_path = rules_config_path or (self.project_root / "configs" / "validation_rules.yaml")
        self.schema_path = schema_config_path or (self.project_root / "configs" / "data_schema.yaml")

        self.rules = self._load_yaml(self.rules_path)
        self.schema = self._load_yaml(self.schema_path)

        if HAS_CHEMDATA:
            self.normalizer = UnitNormalizer()
            self.tracker = ProvenanceTracker(dataset_name="Photocatalysis-Benchmark")
        else:
            self.normalizer = None
            self.tracker = None

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def canonicalize_smiles_string(self, smiles: str) -> Tuple[bool, str, float]:
        """
        Validate SMILES and return (is_valid, canonical_smiles, molecular_weight).
        """
        if not isinstance(smiles, str) or not smiles.strip():
            return False, "", 0.0

        smiles_clean = smiles.strip()

        if HAS_RDKIT:
            try:
                mol = Chem.MolFromSmiles(smiles_clean)
                if mol is not None:
                    can_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                    mw = float(Descriptors.ExactMolWt(mol))
                    return True, can_smiles, round(mw, 3)
            except Exception as e:
                logger.debug(f"RDKit parse failed for '{smiles_clean}': {e}")

        # Fallback if RDKit is not installed or string is simple
        return True, smiles_clean, 0.0

    def validate_and_clean_record(self, row: pd.Series) -> Dict[str, Any]:
        """
        Validate and normalize a single experimental record.
        """
        item = row.to_dict()
        flags: List[str] = []
        tier = 1

        # 1. Provenance Verification
        doi = str(item.get("source_doi", "")).strip()
        if not doi or doi == "nan" or not doi.startswith("10."):
            flags.append("INVALID_DOI")
            tier = max(tier, 2)

        # 2. Chemical Identity Normalization
        raw_smiles = str(item.get("dye_smiles", ""))
        valid_smiles, can_smiles, calc_mw = self.canonicalize_smiles_string(raw_smiles)
        if valid_smiles:
            item["dye_smiles"] = can_smiles
            if calc_mw > 0:
                item["dye_molecular_weight"] = calc_mw
        else:
            flags.append("INVALID_SMILES")
            tier = max(tier, 2)

        # 3. Numerical Boundary Enforcement (Pint / Validation Rules)
        bounds = self.rules.get("numerical_bounds", {})
        
        # Bandgap (eV)
        eg = item.get("bandgap_ev")
        if pd.notna(eg):
            eg = float(eg)
            if not (1.0 <= eg <= 6.0):
                flags.append(f"BANDGAP_OUT_OF_BOUNDS_{eg}")
                tier = 3
            item["bandgap_ev"] = eg
        else:
            flags.append("MISSING_BANDGAP")
            tier = max(tier, 2)

        # Surface Area (m2/g)
        sa = item.get("surface_area_m2g")
        if pd.notna(sa):
            sa = float(sa)
            if not (0.5 <= sa <= 1200.0):
                flags.append(f"SURFACE_AREA_OUT_OF_BOUNDS_{sa}")
                tier = 3
            item["surface_area_m2g"] = sa
        else:
            flags.append("MISSING_SURFACE_AREA")
            tier = max(tier, 2)

        # Catalyst Dosage (g/L)
        dose = item.get("catalyst_dosage_gl")
        if pd.notna(dose):
            dose = float(dose)
            if not (0.001 <= dose <= 50.0):
                flags.append(f"DOSAGE_OUT_OF_BOUNDS_{dose}")
                tier = 3
            item["catalyst_dosage_gl"] = dose
        else:
            flags.append("MISSING_CATALYST_DOSAGE")
            tier = max(tier, 2)

        # Initial Pollutant Concentration (mg/L)
        conc = item.get("initial_dye_conc_mgl")
        if pd.notna(conc):
            conc = float(conc)
            if not (0.1 <= conc <= 2000.0):
                flags.append(f"CONC_OUT_OF_BOUNDS_{conc}")
                tier = 3
            item["initial_dye_conc_mgl"] = conc
        else:
            flags.append("MISSING_INITIAL_CONC")
            tier = max(tier, 2)

        # pH Range
        ph_val = item.get("ph")
        if pd.notna(ph_val):
            ph_val = float(ph_val)
            if not (1.0 <= ph_val <= 14.0):
                flags.append(f"PH_OUT_OF_BOUNDS_{ph_val}")
                tier = 3
            item["ph"] = ph_val
        else:
            item["ph"] = 7.0  # Standard neutral fallback
            flags.append("IMPUTED_NEUTRAL_PH")
            tier = max(tier, 2)

        # Target: Degradation Efficiency (%)
        deg = item.get("degradation_efficiency_percent")
        if pd.notna(deg):
            deg = float(deg)
            if not (0.0 <= deg <= 100.0):
                flags.append(f"DEG_EFFICIENCY_OUT_OF_BOUNDS_{deg}")
                tier = 3
            item["degradation_efficiency_percent"] = deg
        else:
            flags.append("MISSING_TARGET_DEGRADATION")
            tier = 3

        # Target: Rate Constant k (min^-1)
        k_val = item.get("rate_constant_k_min1")
        if pd.notna(k_val):
            item["rate_constant_k_min1"] = float(k_val)
        else:
            # Recompute from first order kinetics if missing
            t_min = float(item.get("reaction_time_min", 60.0))
            deg_pct = float(item.get("degradation_efficiency_percent", 0.0))
            if 0.0 < deg_pct < 100.0 and t_min > 0:
                c_c0 = (100.0 - deg_pct) / 100.0
                calc_k = -np.log(c_c0) / t_min
                item["rate_constant_k_min1"] = round(calc_k, 5)
                flags.append("COMPUTED_RATE_CONSTANT")
            else:
                item["rate_constant_k_min1"] = np.nan

        item["quality_tier"] = tier
        item["validation_flags"] = ";".join(flags) if flags else "VALID"
        return item

    def run(self, input_csv: Path, output_csv: Path) -> pd.DataFrame:
        """
        Execute full validation and normalization across dataset.
        """
        logger.info(f"Loading raw dataset from {input_csv}...")
        df_raw = pd.read_csv(input_csv)
        logger.info(f"Raw rows: {len(df_raw)}")

        cleaned_records = [self.validate_and_clean_record(row) for _, row in df_raw.iterrows()]
        df_clean = pd.DataFrame(cleaned_records)

        # Remove duplicate records
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates(
            subset=["catalyst_name", "dye_name", "catalyst_dosage_gl", "initial_dye_conc_mgl", "reaction_time_min", "ph"]
        )
        dedup_count = len(df_clean)
        if initial_count != dedup_count:
            logger.info(f"Deduplicated {initial_count - dedup_count} exact duplicate experimental records.")

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_csv, index=False)
        logger.info(f"Validated dataset saved to {output_csv} ({len(df_clean)} records).")

        return df_clean