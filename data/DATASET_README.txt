Dataset of the conference paper
"AI-Enabled Asset-Level Digital Twin for Industrial Load Profiling, Short-Term Load Forecasting, and Anomaly Detection"

Source
  C. Flynn, T. Murphy, J. Walsh, and D. Riordan, "An open industrial energy dataset with asset-level
  measurements and high-coverage 15-minute aggregates from a manufacturing facility," Data, vol. 11, no. 5,
  p. 101, 2026. License: CC BY 4.0. The data are redistributed here in reduced form under the same license.

Assets used in the paper (Table I)
  p_c     Press drive (production)
  ahu_a   Air-handling unit (HVAC)
  comp_b  Air compressor
  mh_a    Material handling
  Period: 31 December 2024 to 31 December 2025 (UTC)

Files
  four_assets_15min.csv / .parquet  Reliable three-phase-total 15-minute windows of the four assets (108,740 rows)
      AssetId, window_start_utc, Phase (ALL = three-phase total), IsReliableWindow (only reliable windows kept), Demand_kW
  four_assets_hourly.csv            Hourly mean demand per asset, computed from the 15-minute file (27,215 asset-hours)
      AssetId, hour_utc, Demand_kW_hourly_mean

Check against the paper
  27,215 asset-hours and 1,154 asset-days (Section III-A); 1,096 days with all 24 hours (anomaly study);
  days per asset: p_c 223, ahu_a 345, comp_b 223, mh_a 363 (Table I).
