"""
Output Management System
Handles organized file management for all generated outputs including research, analysis, and content.
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutputManager:
    """Manages organized output files for the pitch deck generation system."""
    
    def __init__(self, base_output_dir: str = "outputs"):
        """
        Initialize the output manager.
        
        Args:
            base_output_dir: Base directory for all outputs
        """
        self.base_output_dir = Path(base_output_dir)
        self.setup_directories()
    
    def setup_directories(self):
        """Create the organized directory structure."""
        directories = [
            self.base_output_dir,
            self.base_output_dir / "research",
            self.base_output_dir / "knowledge_analysis", 
            self.base_output_dir / "content",
            self.base_output_dir / "presentations",
            self.base_output_dir / "reports",
            self.base_output_dir / "logs"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    
    def save_research_output(self, company_name: str, research_data: str, startup_data: Dict[str, Any]) -> str:
        """
        Save research output in organized format.
        
        Args:
            company_name: Name of the company
            research_data: Research output data
            startup_data: Original startup data
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_{self._sanitize_filename(company_name)}_{timestamp}.txt"
        filepath = self.base_output_dir / "research" / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Market Research Report: {company_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Company Information\n")
                f.write(json.dumps(startup_data, indent=2))
                f.write("\n\n## Research Results\n")
                f.write(research_data)
            
            logger.info(f"Research output saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving research output: {e}")
            return ""
    
    def save_knowledge_analysis(self, company_name: str, analysis_data: str, 
                               company_exists: bool, startup_data: Dict[str, Any]) -> str:
        """
        Save knowledge analysis output.
        
        Args:
            company_name: Name of the company
            analysis_data: Knowledge analysis output
            company_exists: Whether company was found in database
            startup_data: Original startup data
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"knowledge_analysis_{self._sanitize_filename(company_name)}_{timestamp}.txt"
        filepath = self.base_output_dir / "knowledge_analysis" / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Knowledge Analysis Report: {company_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Company in Database: {'Yes' if company_exists else 'No'}\n\n")
                f.write("## Company Information\n")
                f.write(json.dumps(startup_data, indent=2))
                f.write("\n\n## Knowledge Analysis Results\n")
                f.write(analysis_data)
            
            logger.info(f"Knowledge analysis saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving knowledge analysis: {e}")
            return ""
    
    def save_content_output(self, company_name: str, content_data: str, startup_data: Dict[str, Any]) -> str:
        """
        Save generated content output.
        
        Args:
            company_name: Name of the company
            content_data: Generated content
            startup_data: Original startup data
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"content_{self._sanitize_filename(company_name)}_{timestamp}.txt"
        filepath = self.base_output_dir / "content" / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# Generated Content: {company_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("## Company Information\n")
                f.write(json.dumps(startup_data, indent=2))
                f.write("\n\n## Generated Pitch Deck Content\n")
                f.write(content_data)
            
            logger.info(f"Content output saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving content output: {e}")
            return ""
    
    def save_presentation(self, company_name: str, ppt_source_path: str) -> str:
        """
        Move and organize PowerPoint presentation file.
        
        Args:
            company_name: Name of the company
            ppt_source_path: Source path of the PowerPoint file
            
        Returns:
            New organized path
        """
        if not os.path.exists(ppt_source_path):
            logger.error(f"Source PowerPoint file not found: {ppt_source_path}")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pitch_deck_{self._sanitize_filename(company_name)}_{timestamp}.pptx"
        filepath = self.base_output_dir / "presentations" / filename
        
        try:
            import shutil
            shutil.move(ppt_source_path, filepath)
            logger.info(f"Presentation moved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error moving presentation: {e}")
            return ppt_source_path  # Return original path if move fails
    
    def save_comprehensive_report(self, company_name: str, report_data: Dict[str, Any]) -> str:
        """
        Save comprehensive workflow report.
        
        Args:
            company_name: Name of the company
            report_data: Complete report data
            
        Returns:
            Path to saved report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_report_{self._sanitize_filename(company_name)}_{timestamp}.json"
        filepath = self.base_output_dir / "reports" / filename
        
        try:
            # Add metadata to report
            enhanced_report = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "company_name": company_name,
                    "workflow_version": "1.0",
                    "output_directory": str(self.base_output_dir)
                },
                "data": report_data
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(enhanced_report, f, indent=2)
            
            logger.info(f"Comprehensive report saved to: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error saving comprehensive report: {e}")
            return ""
    
    def get_company_outputs(self, company_name: str) -> Dict[str, list]:
        """
        Get all outputs for a specific company.
        
        Args:
            company_name: Name of the company
            
        Returns:
            Dictionary with lists of files by category
        """
        sanitized_name = self._sanitize_filename(company_name)
        outputs = {
            "research": [],
            "knowledge_analysis": [],
            "content": [],
            "presentations": [],
            "reports": []
        }
        
        for category in outputs.keys():
            category_dir = self.base_output_dir / category
            if category_dir.exists():
                pattern = f"*{sanitized_name}*"
                files = list(category_dir.glob(pattern))
                outputs[category] = [str(f) for f in sorted(files, key=os.path.getctime, reverse=True)]
        
        return outputs
    
    def get_latest_report(self, company_name: Optional[str] = None) -> Optional[str]:
        """
        Get the path to the latest comprehensive report.
        
        Args:
            company_name: Optional company name to filter by
            
        Returns:
            Path to latest report or None
        """
        reports_dir = self.base_output_dir / "reports"
        
        if company_name:
            pattern = f"*{self._sanitize_filename(company_name)}*.json"
        else:
            pattern = "comprehensive_report_*.json"
        
        report_files = list(reports_dir.glob(pattern))
        
        if report_files:
            latest = max(report_files, key=os.path.getctime)
            return str(latest)
        
        return None
    
    def cleanup_old_files(self, days_old: int = 30):
        """
        Clean up files older than specified days.
        
        Args:
            days_old: Number of days after which files should be cleaned up
        """
        import time
        
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        cleaned_count = 0
        
        for category_dir in [self.base_output_dir / cat for cat in ["research", "knowledge_analysis", "content", "presentations", "reports"]]:
            if category_dir.exists():
                for file_path in category_dir.iterdir():
                    if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                            logger.info(f"Cleaned up old file: {file_path}")
                        except Exception as e:
                            logger.error(f"Error cleaning up {file_path}: {e}")
        
        logger.info(f"Cleanup completed. Removed {cleaned_count} old files.")
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to be filesystem-safe.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        import re
        # Remove or replace invalid characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'\s+', '_', sanitized)  # Replace spaces with underscores
        sanitized = sanitized.strip('._')  # Remove leading/trailing dots and underscores
        return sanitized[:50]  # Limit length
