#!/usr/bin/env python3
"""
RAG Vector Database for Bill Fanter Trading Methodology.

Uses ChromaDB for semantic search over:
- Transcript segments
- Trading signals and levels
- Chart analysis context
- Educational insights

Install: pip install chromadb sentence-transformers
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAG_DIR = DATA_DIR / "rag_pipeline"
TRANSCRIPTS_DIR = DATA_DIR / "transcriber" / "transcripts"
SIGNALS_DIR = DATA_DIR / "transcriber" / "signals"
ALIGNED_DIR = RAG_DIR / "aligned"
CHROMA_DIR = RAG_DIR / "chroma_db"


class BillFanterRAG:
    """
    RAG system for Bill Fanter's trading methodology.

    Stores and retrieves:
    - Transcript segments with timestamps
    - Trade signals and levels
    - Chart context from aligned frames
    - Educational insights
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or CHROMA_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Lazy load ChromaDB
        self._client = None
        self._collection = None

    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings

                self._client = chromadb.PersistentClient(
                    path=str(self.persist_dir),
                    settings=Settings(anonymized_telemetry=False),
                )
            except ImportError:
                raise RuntimeError(
                    "ChromaDB not installed. Run: pip install chromadb sentence-transformers"
                )

        return self._client

    def _get_collection(self):
        """Get or create the main collection."""
        if self._collection is None:
            client = self._get_client()

            # Use sentence-transformers embedding
            self._collection = client.get_or_create_collection(
                name="bill_fanter_methodology",
                metadata={"description": "Bill Fanter trading methodology RAG"},
            )

        return self._collection

    def _generate_id(self, text: str, metadata: dict) -> str:
        """Generate unique ID for a document."""
        # Include more context for uniqueness
        content = (
            f"{metadata.get('video_id', '')}"
            f"{metadata.get('timestamp', '')}"
            f"{metadata.get('type', '')}"
            f"{metadata.get('symbol', '')}"
            f"{text}"  # Full text, not truncated
        )
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def add_transcript_segments(self, video_id: str) -> int:
        """
        Add transcript segments to the database.

        Returns number of segments added.
        """
        transcript_path = TRANSCRIPTS_DIR / f"{video_id}.json"
        if not transcript_path.exists():
            print(f"No transcript found for {video_id}")
            return 0

        with open(transcript_path) as f:
            data = json.load(f)

        segments = data.get("segments", [])
        if not segments:
            print(f"No segments in transcript for {video_id}")
            return 0

        collection = self._get_collection()

        documents = []
        metadatas = []
        ids = []

        for seg in segments:
            text = seg.get("text", "").strip()
            if len(text) < 20:  # Skip very short segments
                continue

            metadata = {
                "type": "transcript_segment",
                "video_id": video_id,
                "timestamp": seg.get("start", 0),
                "end_time": seg.get("end", 0),
            }

            doc_id = self._generate_id(text, metadata)

            documents.append(text)
            metadatas.append(metadata)
            ids.append(doc_id)

        if documents:
            # Add in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_meta = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]

                collection.upsert(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids,
                )

        print(f"Added {len(documents)} transcript segments for {video_id}")
        return len(documents)

    def add_trading_signals(self, video_id: str) -> int:
        """
        Add extracted trading signals to the database.

        Returns number of signals added.
        """
        signal_path = SIGNALS_DIR / f"{video_id}_signals.json"
        if not signal_path.exists():
            print(f"No signals found for {video_id}")
            return 0

        with open(signal_path) as f:
            signals = json.load(f)

        collection = self._get_collection()

        documents = []
        metadatas = []
        ids = []

        # Add trade signals
        for signal in signals.get("trade_signals", []):
            # Create searchable text
            text = (
                f"Trade signal for {signal.get('symbol', 'unknown')}: "
                f"{signal.get('direction', '')} {signal.get('entry_type', '')} "
                f"at {signal.get('zone_type', '')} zone level {signal.get('zone_level', '')}. "
                f"Entry: {signal.get('entry_price', '')}, "
                f"Stop: {signal.get('stop_loss', '')}, "
                f"Target: {signal.get('target', '')}. "
                f"Reasoning: {signal.get('reasoning', '')}"
            )

            metadata = {
                "type": "trade_signal",
                "video_id": video_id,
                "symbol": signal.get("symbol", ""),
                "direction": signal.get("direction", ""),
                "entry_type": signal.get("entry_type", ""),
                "confidence": signal.get("confidence", ""),
            }

            doc_id = self._generate_id(text, metadata)

            documents.append(text)
            metadatas.append(metadata)
            ids.append(doc_id)

        # Add zone levels
        for zone in signals.get("zone_levels", []):
            text = (
                f"Zone level for {zone.get('symbol', 'unknown')}: "
                f"{zone.get('zone_type', '')} zone from {zone.get('zone_low', '')} "
                f"to {zone.get('zone_high', '')} on {zone.get('timeframe', '')} timeframe. "
                f"Freshness: {zone.get('freshness', '')}. "
                f"Context: {zone.get('context', '')}"
            )

            metadata = {
                "type": "zone_level",
                "video_id": video_id,
                "symbol": zone.get("symbol", ""),
                "zone_type": zone.get("zone_type", ""),
                "timeframe": zone.get("timeframe", ""),
            }

            doc_id = self._generate_id(text, metadata)

            documents.append(text)
            metadatas.append(metadata)
            ids.append(doc_id)

        # Add key insights
        for insight in signals.get("key_insights", []):
            text = f"Trading insight from Bill Fanter: {insight}"

            metadata = {
                "type": "insight",
                "video_id": video_id,
            }

            doc_id = self._generate_id(text, metadata)

            documents.append(text)
            metadatas.append(metadata)
            ids.append(doc_id)

        # Add summary
        summary = signals.get("summary", "")
        if summary:
            text = f"Video summary: {summary}"

            metadata = {
                "type": "video_summary",
                "video_id": video_id,
            }

            doc_id = self._generate_id(text, metadata)

            documents.append(text)
            metadatas.append(metadata)
            ids.append(doc_id)

        if documents:
            # Deduplicate by ID
            seen_ids = set()
            deduped_docs = []
            deduped_meta = []
            deduped_ids = []
            for i, doc_id in enumerate(ids):
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    deduped_docs.append(documents[i])
                    deduped_meta.append(metadatas[i])
                    deduped_ids.append(doc_id)

            collection.upsert(
                documents=deduped_docs,
                metadatas=deduped_meta,
                ids=deduped_ids,
            )

        print(f"Added {len(deduped_ids) if documents else 0} signals/insights for {video_id}")
        return len(deduped_ids) if documents else 0

    def add_aligned_frames(self, video_id: str) -> int:
        """
        Add aligned frame-context pairs to the database.

        Returns number of pairs added.
        """
        aligned_path = ALIGNED_DIR / f"{video_id}_aligned.json"
        if not aligned_path.exists():
            print(f"No aligned data found for {video_id}")
            return 0

        with open(aligned_path) as f:
            data = json.load(f)

        collection = self._get_collection()

        documents = []
        metadatas = []
        ids = []

        for pair in data.get("pairs", []):
            context = pair.get("context_text", "").strip()
            if len(context) < 20:
                continue

            # Enrich with signal info
            signal_text = ""
            for sig in pair.get("relevant_signals", []):
                signal_text += (
                    f" [{sig.get('symbol', '')} {sig.get('type', '')}: "
                    f"{sig.get('direction', '')} at {sig.get('zone_level', '')}]"
                )

            text = f"Chart context at {pair.get('timestamp', 0)}s: {context}{signal_text}"

            metadata = {
                "type": "frame_context",
                "video_id": video_id,
                "timestamp": pair.get("timestamp", 0),
                "frame_path": pair.get("frame_path", ""),
                "keyword": pair.get("keyword", ""),
                "extraction_method": pair.get("extraction_method", ""),
            }

            doc_id = self._generate_id(text, metadata)

            documents.append(text)
            metadatas.append(metadata)
            ids.append(doc_id)

        if documents:
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
            )

        print(f"Added {len(documents)} frame contexts for {video_id}")
        return len(documents)

    def index_all_videos(self) -> dict:
        """
        Index all available data into the database.

        Returns stats about what was indexed.
        """
        stats = {
            "transcripts": 0,
            "signals": 0,
            "frames": 0,
            "videos_processed": [],
        }

        # Find all videos with data
        video_ids = set()

        for path in TRANSCRIPTS_DIR.glob("*.json"):
            video_ids.add(path.stem)

        for path in SIGNALS_DIR.glob("*_signals.json"):
            video_ids.add(path.stem.replace("_signals", ""))

        for path in ALIGNED_DIR.glob("*_aligned.json"):
            video_ids.add(path.stem.replace("_aligned", ""))

        print(f"Found {len(video_ids)} videos to index...")

        for video_id in sorted(video_ids):
            print(f"\nIndexing {video_id}...")

            stats["transcripts"] += self.add_transcript_segments(video_id)
            stats["signals"] += self.add_trading_signals(video_id)
            stats["frames"] += self.add_aligned_frames(video_id)
            stats["videos_processed"].append(video_id)

        # Save stats
        stats["indexed_at"] = datetime.now().isoformat()
        stats_path = self.persist_dir / "index_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*50}")
        print(f"Indexing complete!")
        print(f"  Transcript segments: {stats['transcripts']}")
        print(f"  Signals/insights: {stats['signals']}")
        print(f"  Frame contexts: {stats['frames']}")
        print(f"  Total documents: {self.get_collection_count()}")
        print(f"{'='*50}")

        return stats

    def get_collection_count(self) -> int:
        """Get total number of documents in collection."""
        return self._get_collection().count()

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        doc_type: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> list[dict]:
        """
        Query the RAG database.

        Args:
            query_text: Search query
            n_results: Number of results to return
            doc_type: Filter by document type (transcript_segment, trade_signal, etc.)
            symbol: Filter by stock symbol

        Returns:
            List of matching documents with metadata
        """
        collection = self._get_collection()

        # Build where clause
        where = None
        if doc_type or symbol:
            conditions = []
            if doc_type:
                conditions.append({"type": doc_type})
            if symbol:
                conditions.append({"symbol": symbol.upper()})

            if len(conditions) == 1:
                where = conditions[0]
            else:
                where = {"$and": conditions}

        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                })

        return formatted

    def get_trading_context(
        self,
        symbol: str,
        include_zones: bool = True,
        include_insights: bool = True,
    ) -> dict:
        """
        Get all relevant trading context for a symbol.

        Returns comprehensive context for the trading bot.
        """
        context = {
            "symbol": symbol,
            "zones": [],
            "signals": [],
            "insights": [],
            "recent_mentions": [],
        }

        # Get zone levels
        if include_zones:
            zone_results = self.query(
                f"{symbol} zone level supply demand",
                n_results=10,
                symbol=symbol,
            )
            context["zones"] = [r for r in zone_results if r["metadata"].get("type") == "zone_level"]

        # Get trade signals
        signal_results = self.query(
            f"{symbol} trade signal entry",
            n_results=10,
            symbol=symbol,
        )
        context["signals"] = [r for r in signal_results if r["metadata"].get("type") == "trade_signal"]

        # Get general insights
        if include_insights:
            insight_results = self.query(
                f"trading insight strategy",
                n_results=10,
                doc_type="insight",
            )
            context["insights"] = insight_results

        # Get recent transcript mentions
        mention_results = self.query(
            f"{symbol}",
            n_results=5,
            doc_type="transcript_segment",
        )
        context["recent_mentions"] = mention_results

        return context


def main():
    """Test RAG database."""
    print("Initializing RAG database...")
    rag = BillFanterRAG()

    # Index all available data
    stats = rag.index_all_videos()

    # Test queries
    print("\n\n=== TEST QUERIES ===\n")

    # Query 1: General trading methodology
    print("Query: 'supply and demand zone trading'")
    results = rag.query("supply and demand zone trading", n_results=3)
    for r in results:
        print(f"  [{r['metadata'].get('type')}] {r['text'][:100]}...")
    print()

    # Query 2: Specific symbol
    print("Query: 'NVDA levels' with symbol filter")
    results = rag.query("NVDA levels", n_results=3, symbol="NVDA")
    for r in results:
        print(f"  [{r['metadata'].get('type')}] {r['text'][:100]}...")
    print()

    # Query 3: Get full trading context
    print("Getting trading context for TSLA...")
    context = rag.get_trading_context("TSLA")
    print(f"  Zones: {len(context['zones'])}")
    print(f"  Signals: {len(context['signals'])}")
    print(f"  Insights: {len(context['insights'])}")
    print(f"  Mentions: {len(context['recent_mentions'])}")


if __name__ == "__main__":
    main()
