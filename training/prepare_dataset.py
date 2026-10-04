import os
import sys

# Ensure root workspace directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.helpers import save_json_file


def create_sample_news_training_dataset():
    """
    Generates a structured news summarization dataset formatted for instruction fine-tuning.
    """
    dataset = [
        {
            "id": "train_001",
            "article": "Title: Semiconductor Giant Unveils 2nm Chip Architecture\nContent: Global Foundry Corp has announced a breakthrough in 2-nanometer semiconductor fabrication technology. The new transistors promise 30% higher energy efficiency and 20% speed increases for AI accelerators and mobile processors. Mass production is slated for late 2026.",
            "summary": "Title: Semiconductor Giant Unveils 2nm Chip Architecture\nSummary: Global Foundry Corp unveiled a 2nm semiconductor manufacturing process delivering 30% higher energy efficiency and 20% faster speeds. Mass production is scheduled for late 2026.\nKey Points:\n- 2nm gate-all-around transistor architecture\n- 30% improvement in power efficiency\n- Targeted at AI accelerators and mobile processors\nWhy It Matters: Enables faster, more power-efficient hardware for mobile devices and cloud AI infrastructure.\nTopics: Technology, Hardware, AI"
        },
        {
            "id": "train_002",
            "article": "Title: European Union Passes Comprehensive Artificial Intelligence Act\nContent: The EU Parliament voted to approve landmark regulation classifying AI tools by risk levels. High-risk application areas like biometric identification face stringent audits, while foundational models must disclose training data summaries.",
            "summary": "Title: EU Passes Landmark Artificial Intelligence Act\nSummary: The European Union has formally passed regulatory measures categorizing AI models by risk severity. High-risk systems must undergo mandatory compliance audits and data disclosures.\nKey Points:\n- Risk-based regulatory framework for AI\n- Strict audits for biometric surveillance\n- Data transparency requirements for foundational models\nWhy It Matters: Sets global regulatory precedent for governance, safety compliance, and commercial AI deployments.\nTopics: Policy, Governance, AI"
        },
        {
            "id": "train_003",
            "article": "Title: NASA Telescope Detects Water Vapor on Earth-sized Exoplanet\nContent: Astronomers using space observatory instruments detected atmospheric water vapor signatures on exoplanet K2-18b, located 120 light-years away in the habitable zone of a cool dwarf star.",
            "summary": "Title: NASA Telescope Finds Water Vapor Signature on Exoplanet\nSummary: Deep-space observations identified atmospheric water vapor signatures on exoplanet K2-18b, situated within its star's habitable temperature zone.\nKey Points:\n- Atmospheric water vapor detected 120 light-years away\n- Located in star's circumstellar habitable zone\n- High potential candidate for further atmospheric study\nWhy It Matters: Expands astronomical understanding of atmospheric compositions and habitability conditions beyond the solar system.\nTopics: Science, Space, Astronomy"
        },
        {
            "id": "train_004",
            "article": "Title: Central Banks Test Interbank Digital Currency Settlement Network\nContent: A coalition of international central banks completed pilot testing for a cross-border central bank digital currency (CBDC) platform, reducing transaction settlement times from 3 days to under 10 seconds.",
            "summary": "Title: Central Banks Complete Cross-Border CBDC Pilot\nSummary: International financial authorities successfully executed a digital currency trial, accelerating cross-border payments from days to seconds.\nKey Points:\n- Cross-border settlement times reduced to 10 seconds\n- Multi-currency digital token clearing framework\n- Enhanced security and fee reduction for international trade\nWhy It Matters: Modernizes international financial clearance mechanisms and reduces currency transfer frictions.\nTopics: Finance, Digital Currency, Banking"
        },
        {
            "id": "train_005",
            "article": "Title: Renewable Microgrid Project Deployed Across Rural Districts\nContent: State clean energy initiatives deployed solar-battery microgrids to rural villages, supplying continuous power to 50,000 households previously subject to daily blackouts.",
            "summary": "Title: Solar Microgrid Initiative Powers 50,000 Rural Households\nSummary: A renewable energy infrastructure initiative deployed decentralized solar-battery storage systems, ensuring continuous electricity access across remote districts.\nKey Points:\n- Off-grid solar + storage deployment\n- Reliable electricity for 50,000 homes\n- Eliminates daily regional blackout disruptions\nWhy It Matters: Improves rural quality of life and supports sustainable economic growth without expanding fossil-fuel grids.\nTopics: Energy, Infrastructure, Sustainability"
        }
    ]

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "news_summary_dataset.json"))
    save_json_file(dataset, output_path)
    print(f"[OK] News summarization training dataset created successfully at: {output_path}")
    print(f"[INFO] Total training examples: {len(dataset)}")
    return output_path


if __name__ == "__main__":
    create_sample_news_training_dataset()
