"""
GPT-4o Vision Data Extractor
Extracts text and data from PDFs using OpenAI's GPT-4o Vision model.
Converts PDF pages to images and processes them with AI vision for comprehensive data extraction.
"""

import os
import logging
import base64
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import fitz  # PyMuPDF for better PDF to image conversion
from PIL import Image
import io
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPT4VisionExtractor:
    """Extracts data from PDFs using GPT-4o Vision model."""
    
    def __init__(self, data_folder: str = "Data", output_folder: str = "extracted_texts"):
        """
        Initialize the GPT-4o Vision extractor.
        
        Args:
            data_folder: Path to folder containing PDF files
            output_folder: Path to folder for saving extracted text files
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please create a .env file.")
        self.client = OpenAI(api_key=api_key)
    
    def _extract_company_name_from_filename(self, filename: str) -> str:
        """
        Extracts a clean company name from the filename.
        Example: 'uber-pitch-deck.pdf' -> 'uber'
        """
        # Sanitize and extract the first part of the filename
        base_name = Path(filename).stem
        company_name = re.split(r'[-_]', base_name)[0]
        return company_name.lower()
    
    def _parse_llm_company_name(self, response_text: str) -> Optional[str]:
        """Parses the company name from the LLM's response."""
        match = re.search(r"COMPANY_NAME:\s*([^\n]+)", response_text)
        if match:
            company_name = match.group(1).strip()
            # Sanitize for filename
            sanitized_name = re.sub(r'[<>:"/\\|?*]', '', company_name).lower().replace(' ', '_')
            return sanitized_name
        return None
    
    def pdf_to_images(self, pdf_path: Path, dpi: int = 200) -> List[Image.Image]:
        """
        Convert PDF pages to images using PyMuPDF.
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for image conversion
            
        Returns:
            List of PIL Image objects
        """
        images = []
        
        try:
            # Open PDF with PyMuPDF
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Convert page to image
                mat = fitz.Matrix(dpi/72, dpi/72)  # Scaling factor
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
                
                logger.info(f"Converted page {page_num + 1} of {pdf_path.name}")
            
            pdf_document.close()
            logger.info(f"Successfully converted {len(images)} pages from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error converting PDF {pdf_path}: {e}")
        
        return images
    
    def image_to_base64(self, image: Image.Image, format: str = "PNG") -> str:
        """
        Convert PIL Image to base64 string.
        
        Args:
            image: PIL Image object
            format: Image format (PNG, JPEG)
            
        Returns:
            Base64 encoded image string
        """
        buffer = io.BytesIO()
        
        # Optimize image size for API
        if image.size[0] > 2000 or image.size[1] > 2000:
            image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        
        image.save(buffer, format=format, optimize=True, quality=85)
        buffer.seek(0)
        
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def extract_text_from_image(self, image: Image.Image, page_num: int, filename: str, is_first_page: bool = False) -> Dict[str, Any]:
        """
        Extract text and data from image using GPT-4o Vision.
        
        Args:
            image: PIL Image object
            page_num: Page number for context
            filename: Original filename for context
            
        Returns:
            Dictionary with extracted data
        """
        try:
            # Convert image to base64
            image_base64 = self.image_to_base64(image)
            
            # Prepare the prompt for comprehensive extraction
            prompt = """
            You are an expert data analyst. Analyze the provided image of a pitch deck slide.
            Extract ALL text, data, and visual elements. Be comprehensive and precise.
            Structure the output clearly with headings for text, data, and visual insights.
            This information will be used for RAG, so accuracy is critical.
            """

            if is_first_page:
                prompt = """
                This is the first page of a pitch deck. Your most important task is to identify the company name.
                Return the company name on the very first line, in the format:
                COMPANY_NAME: [The Company Name]

                After that, proceed with the full analysis as follows:
                Extract ALL text, data, and visual elements. Be comprehensive and precise.
                Structure the output clearly with headings for text, data, and visual insights.
                """
            
            # Make API call to GPT-4o Vision
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0.1
            )
            
            extracted_content = response.choices[0].message.content
            
            # Structure the response
            result = {
                "page_number": page_num,
                "filename": filename,
                "extracted_content": extracted_content,
                "timestamp": datetime.now().isoformat(),
                "model": "gpt-4o",
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
            }
            
            logger.info(f"Successfully extracted content from page {page_num} of {filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting from page {page_num} of {filename}: {e}")
            return {
                "page_number": page_num,
                "filename": filename,
                "extracted_content": f"Error: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
    
    def process_pdf_file(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Process a single PDF file with GPT-4o Vision.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with all extracted data
        """
        logger.info(f"Starting processing of {pdf_path.name}")
        
        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)
        
        if not images:
            logger.error(f"No images extracted from {pdf_path.name}")
            return {"error": f"Failed to convert {pdf_path.name} to images"}
        
        extracted_pages = []
        total_tokens = 0
        company_name = None

        # Process the first page to get the company name
        if images:
            logger.info(f"Processing page 1 of {pdf_path.name} to identify company name...")
            first_page_data = self.extract_text_from_image(images[0], 1, pdf_path.name, is_first_page=True)
            extracted_pages.append(first_page_data)
            total_tokens += first_page_data.get('tokens_used', 0)
            
            # Attempt to parse the company name from the LLM response
            company_name = self._parse_llm_company_name(first_page_data.get('extracted_content', ''))
            if company_name:
                logger.info(f"LLM identified company name as: '{company_name}'")
            else:
                logger.warning("LLM did not identify company name. Falling back to filename.")
                company_name = self._extract_company_name_from_filename(pdf_path.name)

        # Process the rest of the pages
        for i, image in enumerate(images[1:], 2):
            page_data = self.extract_text_from_image(image, i, pdf_path.name)
            extracted_pages.append(page_data)
            total_tokens += page_data.get('tokens_used', 0)
            import time
            time.sleep(1) # Respect API rate limits
        
        document_data = {
            "filename": pdf_path.name,
            "company_name": company_name, # Add the identified company name
            "total_pages": len(images),
            "processing_timestamp": datetime.now().isoformat(),
            "total_tokens_used": total_tokens,
            "pages": extracted_pages
        }
        
        logger.info(f"Completed processing {pdf_path.name} - {len(images)} pages, {total_tokens} tokens used")
        return document_data
    
    def save_extracted_data(self, document_data: Dict[str, Any]) -> str:
        """
        Save extracted data to text and JSON files.
        
        Args:
            document_data: Dictionary with extracted document data
            
        Returns:
            Path to saved text file
        """
        filename = document_data["filename"]
        # Use the LLM-identified company name for the filename
        company_name = document_data.get("company_name", self._extract_company_name_from_filename(filename))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save files using the company name
        txt_file = self.output_folder / f"{company_name}_extracted_{timestamp}.txt"
        json_file = self.output_folder / f"{company_name}_extracted_{timestamp}.json"
        
        # Create comprehensive text output
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"# Extracted Data from {filename}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"**Processing Date:** {document_data['processing_timestamp']}\n")
            f.write(f"**Total Pages:** {document_data['total_pages']}\n")
            f.write(f"**Total Tokens Used:** {document_data['total_tokens_used']}\n")
            f.write(f"**Extraction Method:** GPT-4o Vision Model\n\n")
            
            # Add content from each page
            for page_data in document_data["pages"]:
                f.write(f"\n## PAGE {page_data['page_number']}\n")
                f.write("-" * 40 + "\n")
                f.write(page_data["extracted_content"])
                f.write("\n\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("End of Document\n")
        
        # Save as JSON for structured access
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(document_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved extracted data to {txt_file} and {json_file}")
        return str(txt_file)
    
    def process_all_pdfs(self) -> List[str]:
        """
        Process all PDF files in the data folder.
        
        Returns:
            List of paths to generated text files
        """
        if not self.data_folder.exists():
            logger.error(f"Data folder {self.data_folder} does not exist")
            return []
        
        # Find all PDF files
        pdf_files = list(self.data_folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.data_folder}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        generated_files = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}...")
                
                # Process the PDF
                document_data = self.process_pdf_file(pdf_file)
                
                if "error" not in document_data:
                    # Save extracted data
                    txt_file = self.save_extracted_data(document_data)
                    generated_files.append(txt_file)
                else:
                    logger.error(f"Failed to process {pdf_file.name}: {document_data['error']}")
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
        
        logger.info(f"Processing complete. Generated {len(generated_files)} text files.")
        return generated_files
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """
        Get summary of extraction results.
        
        Returns:
            Summary statistics
        """
        txt_files = list(self.output_folder.glob("*_extracted_*.txt"))
        json_files = list(self.output_folder.glob("*_extracted_*.json"))
        
        total_tokens = 0
        processed_files = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    total_tokens += data.get('total_tokens_used', 0)
                    processed_files.append({
                        'filename': data.get('filename', 'unknown'),
                        'pages': data.get('total_pages', 0),
                        'tokens': data.get('total_tokens_used', 0)
                    })
            except Exception as e:
                logger.error(f"Error reading {json_file}: {e}")
        
        return {
            "total_extracted_files": len(txt_files),
            "total_tokens_used": total_tokens,
            "processed_documents": processed_files,
            "output_directory": str(self.output_folder)
        }


def main():
    """Main function to run the extraction process."""
    print("🔍 GPT-4o Vision Data Extractor")
    print("=" * 50)
    
    try:
        # Initialize extractor
        extractor = GPT4VisionExtractor()
        
        # Process all PDFs
        print("📄 Processing PDF files...")
        generated_files = extractor.process_all_pdfs()
        
        if generated_files:
            print(f"\n✅ Successfully processed files!")
            print(f"📁 Generated {len(generated_files)} text files:")
            for file_path in generated_files:
                print(f"   • {file_path}")
            
            # Show summary
            summary = extractor.get_extraction_summary()
            print(f"\n📊 Extraction Summary:")
            print(f"   • Total files processed: {summary['total_extracted_files']}")
            print(f"   • Total tokens used: {summary['total_tokens_used']:,}")
            print(f"   • Output directory: {summary['output_directory']}")
            
        else:
            print("❌ No files were processed successfully.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Main execution error: {e}")


if __name__ == "__main__":
    main() 