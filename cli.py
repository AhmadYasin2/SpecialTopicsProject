#!/usr/bin/env python3
"""
AutoPRISMA CLI - Command Line Interface for Systematic Reviews

Usage:
    python cli.py "Your research question here"
    python cli.py "What are the effects of machine learning in healthcare?" --databases pubmed semantic_scholar
    python cli.py "Climate change impacts on agriculture" --date-from 2020-01-01 --date-to 2024-12-31
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from config import settings
from orchestrator import AutoPRISMAOrchestrator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('autoprisma_cli.log')
        ]
    )


def print_banner() -> None:
    """Print CLI banner."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║          AutoPRISMA - Systematic Review System            ║
║                  Command Line Interface                   ║
╚═══════════════════════════════════════════════════════════╝
    """)


def print_results(result: dict) -> None:
    """Print formatted results."""
    print("\n" + "="*60)
    
    if result["status"] == "success":
        print("✅ REVIEW COMPLETED SUCCESSFULLY")
        print("="*60)
        
        final_state = result.get("final_state", {})
        final_report = final_state.get("final_report")
        
        if final_report:
            print(f"\n📄 Report Location:")
            if hasattr(final_report, 'report_path'):
                print(f"   {final_report.report_path}")
            
            # Summary statistics
            print(f"\n📊 Summary Statistics:")
            if hasattr(final_report, 'included_papers'):
                print(f"   • Included papers: {len(final_report.included_papers)}")
            if hasattr(final_report, 'synthesis') and final_report.synthesis:
                print(f"   • Themes identified: {len(final_report.synthesis.themes)}")
                print(f"   • Research gaps: {len(final_report.synthesis.research_gaps)}")
                print(f"   • Consensus findings: {len(final_report.synthesis.consensus_findings)}")
                print(f"   • Contradictions: {len(final_report.synthesis.contradictions)}")
        
        # PRISMA flow diagram
        flow = final_state.get("prisma_flow")
        if flow:
            print(f"\n🔄 PRISMA Flow:")
            try:
                # Handle both dict and Pydantic object
                if isinstance(flow, dict):
                    print(f"   • Records identified: {flow['identification']['total_records']}")
                    print(f"   • After duplicates removed: {flow['screening']['records_screened']}")
                    print(f"   • Records excluded: {flow['screening']['records_excluded']}")
                    print(f"   • Studies included: {flow['included']['studies_in_synthesis']}")
                else:
                    # Pydantic object
                    print(f"   • Records identified: {flow.identification['total_records']}")
                    print(f"   • After duplicates removed: {flow.screening['records_screened']}")
                    print(f"   • Records excluded: {flow.screening['records_excluded']}")
                    print(f"   • Studies included: {flow.included['studies_in_synthesis']}")
            except (KeyError, AttributeError, TypeError) as e:
                print(f"   (Flow diagram data incomplete)")
        
        # Audit trail
        audit_path = result.get("audit_trail_path")
        if audit_path:
            print(f"\n📋 Audit Trail:")
            print(f"   {audit_path}")
        
        # Report paths
        if final_report and hasattr(final_report, 'metadata'):
            report_paths = final_report.metadata.get("report_paths", {})
            if report_paths:
                print(f"\n📁 Generated Reports:")
                for format_type, path in report_paths.items():
                    print(f"   • {format_type.upper()}: {path}")
    
    else:
        print("❌ REVIEW FAILED")
        print("="*60)
        print(f"\n⚠️  Error: {result.get('error', 'Unknown error')}")
        
        audit_path = result.get("audit_trail_path")
        if audit_path:
            print(f"\n📋 Check audit trail for details:")
            print(f"   {audit_path}")
    
    print("\n" + "="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoPRISMA - Automated PRISMA 2020 Systematic Literature Review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python cli.py "What are the effects of vitamin D on bone health?"
  
  # Search specific databases
  python cli.py "Machine learning in cancer diagnosis" --databases pubmed semantic_scholar
  
  # Search with date range
  python cli.py "COVID-19 treatments" --date-from 2020-01-01 --date-to 2024-12-31
  
  # Enable human-in-the-loop review
  python cli.py "Climate change impacts" --enable-hitl
  
  # Verbose output
  python cli.py "Systematic review topic" --verbose

Available databases: pubmed, semantic_scholar, arxiv, openalex
        """
    )
    
    parser.add_argument(
        "query",
        type=str,
        help="Research question for systematic review"
    )
    
    parser.add_argument(
        "--databases",
        type=str,
        nargs="+",
        default=["semantic_scholar", "arxiv"],
        choices=["pubmed", "semantic_scholar", "arxiv", "openalex"],
        help="Databases to search (default: semantic_scholar arxiv)"
    )
    
    parser.add_argument(
        "--date-from",
        type=str,
        metavar="YYYY-MM-DD",
        help="Start date for literature search (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--date-to",
        type=str,
        metavar="YYYY-MM-DD",
        help="End date for literature search (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--enable-hitl",
        action="store_true",
        help="Enable human-in-the-loop checkpoints for manual review"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    print_banner()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting AutoPRISMA review...")
    logger.info(f"Research question: {args.query}")
    logger.info(f"Databases: {', '.join(args.databases)}")
    
    # Validate date range
    date_range = None
    if args.date_from or args.date_to:
        if not (args.date_from and args.date_to):
            print("⚠️  Error: Both --date-from and --date-to must be provided")
            sys.exit(1)
        
        try:
            datetime.strptime(args.date_from, "%Y-%m-%d")
            datetime.strptime(args.date_to, "%Y-%m-%d")
            date_range = (args.date_from, args.date_to)
            logger.info(f"Date range: {args.date_from} to {args.date_to}")
        except ValueError:
            print("⚠️  Error: Dates must be in YYYY-MM-DD format")
            sys.exit(1)
    
    # Check if Ollama is configured
    if settings.llm_provider == "ollama":
        print(f"\n🤖 Using Ollama with model: {settings.llm_model}")
        print(f"   Server: {settings.ollama_base_url}")
        print(f"   Make sure Ollama is running and the model is pulled!")
        print(f"   Run: ollama pull {settings.llm_model}\n")
    
    print(f"\n🔍 Starting systematic review...")
    print(f"   This may take several minutes depending on the query complexity.\n")
    
    try:
        # Initialize orchestrator and run review
        orchestrator = AutoPRISMAOrchestrator()
        result = orchestrator.run_review_sync(
            research_question=args.query,
            databases=args.databases,
            date_range=date_range,
            enable_hitl=args.enable_hitl
        )
        
        # Print results
        print_results(result)
        
        # Exit with appropriate code
        sys.exit(0 if result["status"] == "success" else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Review interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"\n❌ Unexpected error: {e}")
        print(f"   Check autoprisma_cli.log for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
