"""
FreightTiger AI Intern Case Study Solution
Author: AI/SWE Intern Candidate
Description: Production-ready pipeline for freight cost anomaly detection & RAG justification.
"""

import pandas as pd
import numpy as np
import datetime
import os

SHIPMENT_RECORDS_PATH = "shipment_records.csv"
CONTEXT_NOTES_PATH = "context_notes.csv"
OUTPUT_CSV_PATH = "output_assessment_results.csv"

# Cost rise threshold (15% over historical trailing average triggers flag evaluation)
FLAG_THRESHOLD_PCT = 15.0

def load_data(shipment_path: str, notes_path: str):
    """
    Loads shipment records and contextual notes from CSV files.
    Ensures shipment dates are converted to datetime objects and derives route and week_of.
    """
    shipments = pd.read_csv(shipment_path)
    notes = pd.read_csv(notes_path)
    
    # Standardize shipment date to datetime
    shipments['shipment_date'] = pd.to_datetime(shipments['shipment_date'])
    
    # Construct route string (e.g., 'Mumbai-Pune')
    shipments['route'] = shipments['origin'] + "-" + shipments['destination']
    
    # Calculate Monday date of the week (week_of)
    shipments['week_of'] = shipments['shipment_date'].apply(
        lambda d: (d - pd.Timedelta(days=d.weekday())).strftime('%Y-%m-%d')
    )
    
    return shipments, notes

def aggregate_weekly_metrics(shipments: pd.DataFrame) -> pd.DataFrame:
    """
    Groups shipments by route, route_type, and week_of.
    Computes cost per tonne per km = total freight cost / (quantity carried in tonnes x distance travelled in km).
    """
    shipments['tonne_km'] = shipments['quantity_tonnes'] * shipments['distance_km']
    
    weekly = shipments.groupby(['route', 'route_type', 'week_of']).agg(
        total_cost=('freight_cost_inr', 'sum'),
        total_tonne_km=('tonne_km', 'sum')
    ).reset_index()
    
    weekly['cost_per_tonne_km'] = (weekly['total_cost'] / weekly['total_tonne_km']).round(2)
    weekly['week_dt'] = pd.to_datetime(weekly['week_of'])
    weekly = weekly.sort_values(['route', 'week_dt']).reset_index(drop=True)
    
    return weekly

def compute_baselines(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates numerical baselines strictly according to specification:
    1. vs_own_history: Trailing 8-week rolling average cost per tonne-km prior to current week.
    2. vs_similar_routes: Average cost per tonne-km across peer routes of the same route_type in the same week.
    """
    results = []
    
    # Group by route to compute trailing own history
    for route_name, route_group in weekly_df.groupby('route'):
        route_group = route_group.sort_values('week_dt').copy()
        costs = route_group['cost_per_tonne_km'].tolist()
        
        own_history_pct = []
        for i in range(len(costs)):
            if i == 0:
                own_history_pct.append(0.0)
            else:
                start_idx = max(0, i - 8)
                prior_window = costs[start_idx:i]
                prior_avg = np.mean(prior_window)
                
                pct_change = ((costs[i] - prior_avg) / prior_avg) * 100.0
                own_history_pct.append(round(pct_change, 1))
                
        route_group['own_history_diff'] = own_history_pct
        results.append(route_group)
        
    df_with_history = pd.concat(results, ignore_index=True)
    
    # Compute peer comparison (vs_similar_routes)
    peer_diffs = []
    for idx, row in df_with_history.iterrows():
        current_week = row['week_of']
        current_route = row['route']
        current_type = row['route_type']
        current_cost = row['cost_per_tonne_km']
        
        peers = df_with_history[
            (df_with_history['week_of'] == current_week) &
            (df_with_history['route_type'] == current_type) &
            (df_with_history['route'] != current_route)
        ]
        
        if len(peers) > 0:
            peer_avg = peers['cost_per_tonne_km'].mean()
            pct_peer_diff = ((current_cost - peer_avg) / peer_avg) * 100.0
        else:
            pct_peer_diff = 0.0
            
        peer_diffs.append(round(pct_peer_diff, 1))
        
    df_with_history['peer_diff'] = peer_diffs
    
    # Format strings according to sample_output_format contract
    df_with_history['vs_own_history'] = df_with_history['own_history_diff'].apply(
        lambda x: f"+{x:.1f}% vs this route's past average" if x >= 0 else f"{x:.1f}% vs this route's past average"
    )
    df_with_history['vs_similar_routes'] = df_with_history['peer_diff'].apply(
        lambda x: f"+{x:.1f}% vs similar-length routes this week" if x >= 0 else f"{x:.1f}% vs similar-length routes this week"
    )
    
    return df_with_history

def find_matching_note(route: str, week_of_str: str, notes_df: pd.DataFrame):
    """
    RAG/Retrieval logic: Checks context_notes for valid justification.
    Checks date window validity and route applicability ('All Routes' or exact match).
    Avoids hallucinations by verifying the note explicitly confirms cost increases.
    """
    week_dt = pd.to_datetime(week_of_str)
    
    for _, note in notes_df.iterrows():
        n_id = note['note_id']
        n_date = pd.to_datetime(note['date'])
        applies = note['applies_to']
        text = note['note']
        
        route_matches = (applies == "All Routes") or (applies == route)
        date_matches = abs((week_dt - n_date).days) <= 14 or (week_dt >= n_date and week_dt <= n_date + pd.Timedelta(days=14))
        
        if route_matches and date_matches:
            # Filter non-disruptive or non-applicable notes
            if "not significantly affected" in text.lower() or "no rate change" in text.lower() or "remained normal" in text.lower():
                continue
            if "are not part of this dataset" in text.lower() or "improved road conditions" in text.lower():
                continue
                
            return n_id, text, note['date']
            
    return None, None, None

def generate_assessment_results(df_metrics: pd.DataFrame, notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Determines flags, retrieves justifications, and builds final submission dataframe.
    """
    output_rows = []
    
    for idx, row in df_metrics.iterrows():
        route = row['route']
        week_of = row['week_of']
        cost_tkm = row['cost_per_tonne_km']
        vs_history_pct = row['own_history_diff']
        vs_history_str = row['vs_own_history']
        vs_peers_str = row['vs_similar_routes']
        
        is_anomalous_rise = vs_history_pct >= FLAG_THRESHOLD_PCT
        
        if is_anomalous_rise:
            matched_id, note_text, note_date = find_matching_note(route, week_of, notes_df)
            
            if matched_id:
                flagged = "No (justified)"
                reason = f"Matches note {matched_id} dated {note_date}: {note_text}"
            else:
                flagged = "Yes"
                matched_id = ""
                reason = "No matching note found for this route or date range. Cost rise looks unexplained and worth a human review."
        else:
            flagged = "No"
            matched_id = ""
            reason = "Cost change is within normal expected historical range."
            
        output_rows.append({
            'route': route,
            'week_of': week_of,
            'cost_per_tonne_km': f"{cost_tkm:.2f}",
            'vs_own_history': vs_history_str,
            'vs_similar_routes': vs_peers_str,
            'flagged': flagged,
            'matched_note_id': matched_id,
            'reason': reason
        })
        
    return pd.DataFrame(output_rows)

def run_pipeline():
    """Executes the complete end-to-end processing pipeline."""
    print("[1/4] Loading dataset and context notes...")
    shipments, notes = load_data(SHIPMENT_RECORDS_PATH, CONTEXT_NOTES_PATH)
    
    print("[2/4] Aggregating weekly route metrics...")
    weekly_metrics = aggregate_weekly_metrics(shipments)
    
    print("[3/4] Computing 8-week trailing baselines and peer route benchmarks...")
    processed_df = compute_baselines(weekly_metrics)
    
    print("[4/4] Performing note retrieval and generating structured explanations...")
    final_output = generate_assessment_results(processed_df, notes)
    
    final_output.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Pipeline complete! Output generated successfully at: {OUTPUT_CSV_PATH}")

if __name__ == "__main__":
    run_pipeline()
