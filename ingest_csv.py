import os
import pickle
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 1. LOAD & SUMMARISE ───────────────────────────────────────────────────────

def clean_csv(df):
    """
    Fix columns that should be numeric but are stored as strings.
    Acquisition_Cost: '$16,174.00' -> 16174.00
    Duration: '30 days' -> 30
    """
    df["Acquisition_Cost"] = (
        df["Acquisition_Cost"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df["Duration"] = (
        df["Duration"]
        .str.replace(" days", "", regex=False)
        .astype(int)
    )
    print("Cleaned Acquisition_Cost and Duration columns")
    return df


def build_comparison_chunks(df):
    """
    Build natural language comparison summaries that directly answer
    'which channel has the highest/lowest X' questions.

    These are the chunks BM25 was missing — no single group summary
    says 'highest' or 'lowest', so comparative queries scored poorly.
    Adding explicit comparison sentences fixes this directly.

    We build comparisons for every key metric across two dimensions:
      - Channel_Used (Email, Google Ads, Facebook, etc.)
      - Campaign_Type (Email, Influencer, Social Media, etc.)
    """
    chunks = []
    metrics = {
        "Conversion_Rate": ("conversion rate", "{:.2%}"),
        "ROI":             ("ROI",             "{:.2f}x"),
        "Acquisition_Cost":("acquisition cost","${:,.2f}"),
        "Clicks":          ("clicks",          "{:,.0f}"),
        "Engagement_Score":("engagement score","{:.1f}/10"),
    }
    dimensions = ["Channel_Used", "Campaign_Type"]

    for dim in dimensions:
        dim_label = "channel" if dim == "Channel_Used" else "campaign type"
        grouped = df.groupby(dim).agg({m: "mean" for m in metrics}).reset_index()

        for metric_col, (metric_name, fmt) in metrics.items():
            sorted_asc  = grouped.sort_values(metric_col, ascending=True)
            sorted_desc = grouped.sort_values(metric_col, ascending=False)

            best  = sorted_desc.iloc[0]
            worst = sorted_asc.iloc[0]

            # format all values for the ranking sentence
            ranking = ", ".join(
                f"{row[dim]} ({fmt.format(row[metric_col])})"
                for _, row in sorted_desc.iterrows()
            )

            text = (
                f"Among all {dim_label}s, {best[dim]} has the highest average {metric_name} "
                f"at {fmt.format(best[metric_col])}, while {worst[dim]} has the lowest "
                f"at {fmt.format(worst[metric_col])}. "
                f"Full ranking by {metric_name}: {ranking}."
            )

            chunks.append({
                "text":     text,
                "source":   f"Campaign Data: {dim_label} comparison by {metric_name}",
                "presplit": True,
            })

    print(f"  Created {len(chunks)} comparison summary chunks")
    return chunks


def load_csv(path):
    """
    Load the CSV and create three types of text representations:

    A) Comparison summaries — best/worst/ranking per metric per dimension
       Directly answers 'which channel has the highest X?' questions.

    B) Group summaries — average performance per Campaign_Type + Channel_Used
       Answers strategic questions like 'how did email campaigns perform?'

    C) Individual samples — 300 random rows for row-level detail.
       Answers specific questions like 'tell me about influencer campaigns.'
    """
    df = pd.read_csv(path)
    print(f"Loaded CSV: {df.shape[0]:,} rows x {df.shape[1]} columns")
    df = clean_csv(df)

    pages = []

    # ── A) COMPARISON SUMMARIES ───────────────────────────────────────────────
    print("Creating comparison summary chunks...")
    comparison_chunks = build_comparison_chunks(df)
    pages.extend(comparison_chunks)

    # ── B) GROUP SUMMARIES ────────────────────────────────────────────────────
    print("Creating group summaries...")
    summary_df = (
        df.groupby(["Campaign_Type", "Channel_Used"])
        .agg(
            avg_roi=("ROI", "mean"),
            avg_conversion_rate=("Conversion_Rate", "mean"),
            avg_acquisition_cost=("Acquisition_Cost", "mean"),
            avg_duration=("Duration", "mean"),
            avg_clicks=("Clicks", "mean"),
            avg_impressions=("Impressions", "mean"),
            avg_engagement=("Engagement_Score", "mean"),
            total_campaigns=("Campaign_ID", "count")
        )
        .reset_index()
    )

    for _, row in summary_df.iterrows():
        text = (
            f"{row['Campaign_Type']} campaigns run via {row['Channel_Used']} "
            f"had an average ROI of {row['avg_roi']:.2f}x, "
            f"an average conversion rate of {row['avg_conversion_rate']:.2%}, "
            f"an average acquisition cost of ${row['avg_acquisition_cost']:,.2f}, "
            f"an average duration of {row['avg_duration']:.0f} days, "
            f"an average of {row['avg_clicks']:.0f} clicks, "
            f"{row['avg_impressions']:.0f} impressions, "
            f"and an engagement score of {row['avg_engagement']:.1f}/10. "
            f"This is based on {row['total_campaigns']} campaigns in the dataset."
        )
        pages.append({
            "text":     text,
            "source":   f"Campaign Data: {row['Campaign_Type']} via {row['Channel_Used']}",
            "presplit": True,
        })

    print(f"  Created {len(summary_df)} group summary rows")

    # ── C) INDIVIDUAL SAMPLES ─────────────────────────────────────────────────
    print("Creating individual campaign samples...")
    sample_df = df.sample(n=min(300, len(df)), random_state=42)

    for _, row in sample_df.iterrows():
        text = (
            f"{row['Company']} ran a {row['Campaign_Type']} campaign "
            f"via {row['Channel_Used']} targeting {row['Target_Audience']} "
            f"({row['Customer_Segment']}) in {row['Location']}. "
            f"The campaign ran for {row['Duration']} days and achieved "
            f"an ROI of {row['ROI']:.2f}x, a conversion rate of {row['Conversion_Rate']:.2%}, "
            f"{row['Clicks']:,} clicks, {row['Impressions']:,} impressions, "
            f"an engagement score of {row['Engagement_Score']}/10, "
            f"and an acquisition cost of ${row['Acquisition_Cost']:,.2f}. "
            f"Campaign date: {row['Date']}. Language: {row['Language']}."
        )
        pages.append({
            "text": text,
            "source": f"Campaign Data: {row['Company']} — {row['Campaign_Type']} ({row['Date']})"
        })

    print(f"  Created 300 individual campaign samples")
    print(f"\nTotal pages to chunk: {len(pages)}")
    return pages


# ── 2. CHUNK ──────────────────────────────────────────────────────────────────

def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = []
    for page in pages:
        if page.get("presplit"):
            chunks.append({"text": page["text"], "source": page["source"]})
        else:
            for split in splitter.split_text(page["text"]):
                chunks.append({"text": split, "source": page["source"]})
    print(f"Created {len(chunks)} chunks from CSV")
    return chunks


# ── 3. SAVE ───────────────────────────────────────────────────────────────────

def save_chunks(chunks, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved chunks to {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    csv_path    = "data/campaign_performance.csv"
    output_path = "chunks_csv.pkl"

    if not os.path.exists(csv_path):
        print(f"ERROR: Could not find {csv_path}")
    else:
        pages  = load_csv(csv_path)
        chunks = chunk_pages(pages)
        save_chunks(chunks, output_path)

        if chunks:
            print("\nDone! Preview of first 3 chunks:\n")
            for i, chunk in enumerate(chunks[:3]):
                print(f"--- Chunk {i + 1} ---")
                print(f"Source : {chunk['source']}")
                print(f"Text   : {chunk['text'][:200]}")
                print()
