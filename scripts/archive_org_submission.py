import argparse
import logging
import time
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from waybackpy import WaybackMachineSaveAPI, WaybackMachineCDXServerAPI, exceptions as waybackpy_exceptions
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests


def setup_logging():
    """Configure logging for the script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='archive_submitter.log',
        filemode='a'
    )
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


def create_session():
    """Create a requests session with retry logic"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"
    })
    return session


def normalize_url(url):
    """Normalize a URL to ensure it's properly formatted"""
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        if url.startswith('www.'):
            url = 'https://' + url
        elif url.startswith('foxnews.com/') or url.startswith('www.foxnews.com/'):
            url = 'https://' + url
        else:
            # Handle paths by adding the domain
            if url.startswith('/'):
                url = url[1:]  # Remove leading slash
            url = f"https://www.foxnews.com/{url}"
    
    # Clean up formatting issues
    url = url.replace("//", "/").replace("https:/", "https://").replace("http:/", "http://")
    if url.count("https://") > 1:
        url = url.replace("https://www.foxnews.com/https://", "https://")
    
    # Fix any remaining double slashes (except in protocol)
    parts = url.split("://", 1)
    if len(parts) > 1:
        protocol = parts[0] + "://"
        path = parts[1]
        while "//" in path:
            path = path.replace("//", "/")
        url = protocol + path
    
    return url


def check_archive_exists(url, user_agent):
    """
    Check if a URL is already archived in the Wayback Machine
    
    Args:
        url: URL to check
        user_agent: User agent string to use for the request
        
    Returns:
        Boolean: True if archived, False if not
    """
    try:
        cdx_api = WaybackMachineCDXServerAPI(
            url=url,
            user_agent=user_agent
        )
        
        # Just check if any snapshots exist
        # Since snapshots() returns a generator, we need to check for items differently
        snapshots = cdx_api.snapshots()
        # Try to get the first item from the generator
        try:
            first_snapshot = next(snapshots)
            return True  # If we get here, at least one snapshot exists
        except StopIteration:
            return False  # No snapshots exist
        
    except waybackpy_exceptions.NoCDXRecordFound:
        return False
    except Exception as e:
        logging.warning(f"Error checking archive for {url}: {e}")
        return False  # Assume not archived if we can't check


def submit_url_to_archive(url, user_agent, fast_mode=True):
    """
    Submit a URL to be archived by the Wayback Machine
    
    Args:
        url: URL to archive
        user_agent: User agent string to use for the request
        fast_mode: If True, don't wait for confirmation
        
    Returns:
        Tuple: (success boolean, archive URL or error message)
    """
    try:
        if fast_mode:
            # Use a simple request to submit without waiting for completion
            save_url = f"https://web.archive.org/save/{url}"
            headers = {"User-Agent": user_agent}
            response = requests.get(save_url, headers=headers, timeout=10)
            
            if response.status_code in [200, 301, 302]:
                # Submission initiated successfully
                return True, f"Submitted to {save_url}"
            else:
                return False, f"HTTP error: {response.status_code}"
        else:
            # Use waybackpy to submit and wait for confirmation
            save_api = WaybackMachineSaveAPI(url=url, user_agent=user_agent)
            archived_url = save_api.save()
            return True, archived_url
            
    except Exception as e:
        return False, str(e)


def process_url(args):
    """Process a single URL submission"""
    url, index, user_agent, delay, check_first, fast_mode = args
    
    # Normalize URL
    normalized_url = normalize_url(url)
    
    # First check if the URL is already archived (if requested)
    if check_first:
        is_archived = check_archive_exists(normalized_url, user_agent)
        if is_archived:
            return index, url, normalized_url, "ALREADY_ARCHIVED", None
    
    # Add a delay between requests to be respectful
    time.sleep(delay)
    
    # Submit URL to archive.org
    try:
        success, result = submit_url_to_archive(normalized_url, user_agent, fast_mode)
        
        if success:
            return index, url, normalized_url, "SUBMITTED", result
        else:
            return index, url, normalized_url, "ERROR", result
    except Exception as e:
        return index, url, normalized_url, "EXCEPTION", str(e)


def main(urls, output_file=None, workers=2, delay=5.0, check_first=True, fast_mode=True):
    """Main function to run the URL submission"""
    setup_logging()
    
    # Create session for user agent
    session = create_session()
    user_agent = session.headers["User-Agent"]
    
    logging.info(f"Starting archive submission for {len(urls)} URLs")
    if check_first:
        logging.info("Checking for existing snapshots first")
    if fast_mode:
        logging.info("Fast mode enabled - not waiting for archiving to complete")
    
    # Prepare tasks
    tasks = [(url, i, user_agent, delay, check_first, fast_mode) for i, url in enumerate(urls)]
    
    # Create progress bar
    pbar = tqdm(total=len(urls))
    
    # Process URLs with thread pool
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = executor.map(process_url, tasks)
        
        for result in futures:
            results.append(result)
            pbar.update(1)
    
    pbar.close()
    
    # Summarize results
    submitted_count = sum(1 for _, _, _, status, _ in results if status == "SUBMITTED")
    already_archived = sum(1 for _, _, _, status, _ in results if status == "ALREADY_ARCHIVED")
    error_count = sum(1 for _, _, _, status, _ in results if status in ["ERROR", "EXCEPTION"])
    
    logging.info(f"Completed: {len(results)} URLs processed")
    logging.info(f"Summary: {submitted_count} submitted, {already_archived} already archived, {error_count} errors")
    
    # Save detailed report if requested
    if output_file:
        detailed_results = []
        for index, original_url, normalized_url, status, message in results:
            result_dict = {
                'index': index,
                'original_url': original_url,
                'normalized_url': normalized_url,
                'status': status,
                'message': message
            }
            detailed_results.append(result_dict)
            
        report_df = pd.DataFrame(detailed_results)
        
        # Save as gzipped CSV if output_file ends with .gz
        if output_file.endswith('.gz'):
            report_df.to_csv(output_file, index=False, compression='gzip')
            logging.info(f"Saved compressed report to: {output_file}")
        else:
            report_df.to_csv(output_file, index=False)
            logging.info(f"Saved report to: {output_file}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Quickly submit URLs to the Wayback Machine')
    parser.add_argument('--input', type=str, required=True, help='Input file with URLs (CSV, TXT, or CSV.GZ)')
    parser.add_argument('--url-column', type=str, default='url', help='Column name containing URLs (for CSV input)')
    parser.add_argument('--output', type=str, default='archive_results.csv', help='Output file for results')
    parser.add_argument('--workers', type=int, default=2, help='Number of worker threads (keep low to be respectful)')
    parser.add_argument('--delay', type=float, default=5.0, help='Delay between submissions in seconds')
    parser.add_argument('--no-check', action='store_true', help='Skip checking if URLs are already archived')
    parser.add_argument('--wait', action='store_true', help='Wait for archive confirmation (slower)')
    
    args = parser.parse_args()
    
    # Load URLs from input file
    input_path = Path(args.input)
    urls = []
    
    if input_path.suffix.lower() in ['.csv']:
        # Read from CSV
        df = pd.read_csv(input_path)
        if args.url_column in df.columns:
            urls = df[args.url_column].tolist()
        else:
            raise ValueError(f"Column {args.url_column} not found in CSV file")
    elif input_path.suffix.lower() in ['.gz'] and input_path.stem.endswith('.csv'):
        # Read from gzipped CSV
        df = pd.read_csv(input_path, compression='gzip')
        if args.url_column in df.columns:
            urls = df[args.url_column].tolist()
        else:
            raise ValueError(f"Column {args.url_column} not found in CSV file")
    elif input_path.suffix.lower() in ['.txt']:
        # Read from plain text file (one URL per line)
        with open(input_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError("Unsupported input file format. Please use .csv, .csv.gz, or .txt")
    
    # Filter out empty URLs
    urls = [url for url in urls if url]
    
    # Run the submission process
    main(urls, output_file=args.output, workers=args.workers, delay=args.delay, 
         check_first=not args.no_check, fast_mode=not args.wait)