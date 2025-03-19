import os
import re
import time
import json
import logging
import requests
import pandas as pd
import gzip
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse
from waybackpy import WaybackMachineCDXServerAPI, exceptions as waybackpy_exceptions


def setup_logging():
    """Configure logging for the scraper"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='scraper.log',
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


def get_wayback_snapshots(url, user_agent):
    """
    Use waybackpy to get available snapshots for a URL
    
    Returns a list of snapshots with their timestamps and URLs
    """
    # Clean the URL to prevent formatting issues
    url = url.replace("//", "/").replace("https:/", "https://").replace("http:/", "http://")
    if url.count("https://") > 1:
        url = url.replace("https://www.foxnews.com/https://", "https://")
    
    # Ensure we don't have double slashes elsewhere in the URL
    while "//" in url and not url.startswith("http://") and not "https://" in url:
        url = url.replace("//", "/")
        
    logging.debug(f"Querying Wayback Machine for URL: {url}")
    
    try:
        cdx_api = WaybackMachineCDXServerAPI(
            url=url,
            user_agent=user_agent,
            filter_field="statuscode",
            filter_value=200
        )
        
        # Get snapshots and sort by timestamp (newest first)
        snapshots = cdx_api.snapshots()
        if not snapshots:
            logging.warning(f"No snapshots found for {url}")
            return []
            
        # Sort by timestamp (newest first)
        snapshots.sort(key=lambda x: x.timestamp, reverse=True)
        return snapshots
        
    except waybackpy_exceptions.NoCDXRecordFound:
        logging.warning(f"No CDX record found for {url}")
        return []
    except Exception as e:
        logging.error(f"Error fetching snapshots for {url}: {e}")
        return []


def download_from_wayback(snapshot, session, output_path):
    """
    Download a specific snapshot from the Wayback Machine using waybackpy snapshot
    
    Args:
        snapshot: Waybackpy snapshot object
        session: Requests session
        output_path: Path where the file should be saved
    
    Returns:
        Tuple: (success boolean, archive URL or error message)
    """
    try:
        # Get the archive URL
        wayback_url = snapshot.archive_url
        
        # Download the snapshot
        response = session.get(wayback_url, timeout=30)
        
        if response.status_code != 200:
            return False, f"HTTP error: {response.status_code}"
        
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the content as gzipped file
        with gzip.open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Log the archive information
        logging.info(f"Archive info for {output_path.name}:")
        logging.info(f"  Downloaded from: {wayback_url}")
        logging.info(f"  Original URL: {snapshot.original}")
        logging.info(f"  Archive.org timestamp: {snapshot.timestamp}")
        
        return True, wayback_url
        
    except Exception as e:
        return False, str(e)


def download_article(row, session, output_dir, try_archive=True, archive_timestamp=None):
    """Download a single article and save to file"""
    i, r = row
    
    # Create the output filename with .gz extension
    html_filename = r['html_file']
    if not html_filename.endswith('.gz'):
        html_filename = html_filename + '.gz'
    
    fn = Path(output_dir) / html_filename
    
    # Skip if already downloaded
    if fn.exists():
        # Verify file isn't empty or corrupt (has minimum valid size)
        if fn.stat().st_size > 50:  # Lower threshold for gzipped files
            return i, r['url'], "SKIPPED", None
        else:
            logging.warning(f"Found potentially corrupt file: {fn} ({fn.stat().st_size} bytes)")
            # We'll redownload by continuing execution
    
    # Construct the full URL
    if r['url'].startswith(('http://', 'https://')):
        # URL already has protocol - don't add domain
        logging.info(f"URL already contains protocol: {r['url']}")
        original_url = r['url']  # Use as is
    elif r['url'].startswith('www.foxnews.com/'):
        # URL already has domain but no protocol
        logging.info(f"URL already contains domain: {r['url']}")
        original_url = f"https://{r['url']}"
    else:
        # Standard case - add domain
        # Handle leading slashes to avoid double slashes
        url_path = r['url']
        if url_path.startswith('/'):
            url_path = url_path[1:]  # Remove leading slash
        original_url = f"https://www.foxnews.com/{url_path}"
    
    # Create variants with both HTTP and HTTPS to try before archive.org
    url_variants = [original_url]
    
    # Add alternative protocol version if not already included
    if original_url.startswith('https://'):
        http_url = original_url.replace('https://', 'http://', 1)
        url_variants.append(http_url)
    elif original_url.startswith('http://'):
        https_url = original_url.replace('http://', 'https://', 1)
        url_variants.append(https_url)
    
    logging.info(f"Will try multiple URL variants before fallback: {url_variants}")
    
    # Try all URL variants before falling back to archive.org
    for variant_url in url_variants:
        try:
            logging.info(f"Trying URL variant: {variant_url}")
            response = session.get(variant_url, timeout=10)
            
            if response.status_code == 200:
                # Create parent directory if it doesn't exist
                fn.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the content as gzipped file
                with gzip.open(fn, 'wb') as f:
                    f.write(response.content)
                    
                return i, r['url'], "SUCCESS", None
        except Exception as e:
            logging.info(f"URL variant failed: {variant_url}, Error: {e}")
            # Continue to next variant
    
    # If we get here, all URL variants failed
    if try_archive:
        # Now try archive.org as last resort
        logging.info(f"All direct URL attempts failed, trying archive.org: {r['url']}")
        return try_archive_org_download_with_waybackpy(i, r, original_url, fn, session, archive_timestamp)
    else:
        return i, r['url'], "HTTP_ERROR", "All URL variants failed"


def try_archive_org_download_with_waybackpy(i, r, original_url, output_path, session, archive_timestamp=None):
    """
    Try to download from archive.org using waybackpy
    """
    try:
        # Ensure the original_url doesn't have duplicated domains
        if "foxnews.com/https://www.foxnews.com" in original_url:
            # Fix duplicate domain issue
            corrected_url = original_url.replace("https://www.foxnews.com/https://www.foxnews.com/", "https://www.foxnews.com/")
            logging.warning(f"Fixed duplicate domain in URL: {original_url} -> {corrected_url}")
            original_url = corrected_url
            
        # Try both HTTP and HTTPS versions when querying archive.org
        urls_to_try = [original_url]
        
        # Add alternative protocol version if not already in the list
        if original_url.startswith('https://'):
            http_url = original_url.replace('https://', 'http://', 1)
            urls_to_try.append(http_url)
        elif original_url.startswith('http://'):
            https_url = original_url.replace('http://', 'https://', 1)
            urls_to_try.append(https_url)
            
        logging.info(f"Will try multiple URL variants in Archive.org: {urls_to_try}")
        
        # If a specific timestamp is provided, use it directly
        if archive_timestamp:
            # Try to access the specific snapshot with both URL variants
            for url_variant in urls_to_try:
                try:
                    # Use waybackpy to get the archive URL for the specific timestamp
                    cdx_api = WaybackMachineCDXServerAPI(
                        url=url_variant,
                        user_agent=session.headers["User-Agent"]
                    )
                    
                    # Find the closest snapshot to the requested timestamp
                    snapshots = cdx_api.snapshots()
                    if not snapshots:
                        continue
                        
                    # Sort snapshots by how close they are to the requested timestamp
                    snapshots.sort(key=lambda x: abs(int(x.timestamp) - int(archive_timestamp)))
                    closest_snapshot = snapshots[0]
                    
                    # Download the snapshot
                    success, result = download_from_wayback(closest_snapshot, session, output_path)
                    if success:
                        return i, r['url'], "SUCCESS_ARCHIVE", None
                except Exception as e:
                    logging.warning(f"Error accessing specific timestamp for {url_variant}: {e}")
                    continue
                
            # If specific timestamp fails with all variants, try to find others
            logging.warning(f"Specified archive timestamp failed for all URL variants, trying to find others")
        
        # Try to get available snapshots for both URL variants
        all_snapshots = []
        for url_variant in urls_to_try:
            snapshots = get_wayback_snapshots(url_variant, session.headers["User-Agent"])
            if snapshots:
                logging.info(f"Found {len(snapshots)} snapshots for {url_variant}")
                all_snapshots.extend(snapshots)
        
        # Remove duplicate snapshots (by timestamp)
        seen_timestamps = set()
        unique_snapshots = []
        for snapshot in all_snapshots:
            if snapshot.timestamp not in seen_timestamps:
                seen_timestamps.add(snapshot.timestamp)
                unique_snapshots.append(snapshot)
                
        # Sort by timestamp (newest first)
        unique_snapshots.sort(key=lambda x: x.timestamp, reverse=True)
        
        if unique_snapshots:
            # Try each snapshot in order (newest first) until one works
            attempts = 0
            max_attempts = min(6, len(unique_snapshots))  # Limit total attempts to 6
            
            for snapshot in unique_snapshots[:max_attempts]:
                logging.info(f"Trying snapshot from {snapshot.timestamp} for URL {snapshot.original}")
                success, result = download_from_wayback(snapshot, session, output_path)
                attempts += 1
                
                if success:
                    return i, r['url'], "SUCCESS_ARCHIVE", None
            
            # If we tried all snapshots and none worked
            return i, r['url'], "ARCHIVE_ERROR", f"All {attempts} snapshot attempts failed"
        else:
            return i, r['url'], "ARCHIVE_ERROR", "No snapshots found for any URL variant"
                
    except Exception as e:
        return i, r['url'], "ARCHIVE_EXCEPTION", str(e)


def process_batch(batch, session, output_dir, delay=0.5, try_archive=True, archive_timestamp=None):
    """Process a batch of articles with delay between requests"""
    results = []
    
    for row in batch:
        result = download_article(row, session, output_dir, try_archive, archive_timestamp)
        results.append(result)
        time.sleep(delay)  # Respectful delay between requests
        
    return results


def main(dataframe, output_dir='html', workers=4, batch_size=10, delay=0.5, 
          try_archive=True, archive_timestamp=None, output_report=None):
    """Main function to run the scraper"""
    # Setup
    setup_logging()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check for existing files (both .html and .html.gz formats for compatibility)
    existing_files = set()
    for f in output_path.glob('**/*'):
        if f.is_file():
            existing_files.add(f.name)
            # Also add the non-gzipped version to avoid redownloading
            if f.name.endswith('.gz'):
                existing_files.add(f.name[:-3])
    
    logging.info(f"Found {len(existing_files)} existing files in {output_dir}")
    
    # Add .gz extension to html_file in dataframe if not already present
    dataframe['html_file_gz'] = dataframe['html_file'].apply(
        lambda x: x if x.endswith('.gz') else x + '.gz')
    
    # Count how many we can skip
    skippable = sum(1 for _, row in dataframe.iterrows() 
                   if row['html_file'] in existing_files or row['html_file_gz'] in existing_files)
    
    logging.info(f"Starting scraper for {len(dataframe)} articles ({skippable} can be skipped)")
    
    # Prepare data and session
    session = create_session()
    
    # Create progress bar
    pbar = tqdm(total=len(dataframe))
    
    # Process in batches across multiple threads
    results = []
    
    # Create batches
    batches = []
    for i in range(0, len(dataframe), batch_size):
        batch = list(dataframe.iloc[i:i+batch_size].iterrows())
        batches.append(batch)
    
    # Process batches with thread pool
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(process_batch, batch, session, output_dir, delay, 
                                    try_archive, archive_timestamp)
            futures.append(future)
        
        # Process results as they complete
        for future in futures:
            batch_results = future.result()
            for i, url, status, error in batch_results:
                if status == "SUCCESS":
                    logging.info(f"Downloaded ({i}): {url}")
                elif status == "SUCCESS_ARCHIVE":
                    logging.info(f"Downloaded from archive.org ({i}): {url} (see log for details)")
                elif status == "SKIPPED":
                    logging.debug(f"Skipped ({i}): {url}")
                elif status == "HTTP_ERROR":
                    logging.error(f"HTTP Error ({i}, {error}): {url}")
                elif status == "ARCHIVE_ERROR":
                    logging.error(f"Archive.org Error ({i}): {url} - {error}")
                else:
                    logging.error(f"Exception ({i}, {error}): {url}")
                
                results.append((i, url, status, error))
                pbar.update(1)
                
                if i % 100 == 0 and i > 0:
                    logging.info(f"Progress: {i} articles processed")
    
    pbar.close()
    
    # Summarize results
    success = sum(1 for _, _, status, _ in results if status == "SUCCESS")
    success_archive = sum(1 for _, _, status, _ in results if status == "SUCCESS_ARCHIVE")
    skipped = sum(1 for _, _, status, _ in results if status == "SKIPPED")
    errors = sum(1 for _, _, status, _ in results if status in ["HTTP_ERROR", "EXCEPTION", 
                                                               "ARCHIVE_ERROR", "ARCHIVE_EXCEPTION"])
    
    logging.info(f"Completed: {len(results)} articles processed")
    logging.info(f"Summary: {success} downloaded directly, {success_archive} from archive.org, "
                f"{skipped} skipped, {errors} errors")
                
    # Save detailed report if requested
    if output_report:
        # Create a more detailed report that includes archive.org information
        detailed_results = []
        for i, url, status, error in results:
            result_dict = {
                'index': i,
                'url': url,
                'status': status,
                'error': error
            }
            detailed_results.append(result_dict)
            
        report_df = pd.DataFrame(detailed_results)
        
        # Save as gzipped CSV if output_report ends with .gz
        if output_report.endswith('.gz'):
            report_df.to_csv(output_report, index=False, compression='gzip')
            logging.info(f"Saved compressed report to: {output_report}")
        else:
            report_df.to_csv(output_report, index=False)
            logging.info(f"Saved report to: {output_report}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    # Command line arguments
    parser = argparse.ArgumentParser(description='Fox News Article Scraper with WaybackPy')
    parser.add_argument('--input', type=str, default="articles.csv.gz", help='Input CSV file with article data (supports .csv or .csv.gz)')
    parser.add_argument('--output-dir', type=str, default="html", help='Directory to save HTML files')
    parser.add_argument('--workers', type=int, default=4, help='Number of worker threads')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between requests in seconds')
    parser.add_argument('--force-redownload', action='store_true', help='Force redownload of existing files')
    parser.add_argument('--no-archive', action='store_true', help='Disable fallback to archive.org')
    parser.add_argument('--archive-timestamp', type=str, help='Specific archive.org timestamp (e.g., 20200727022720)')
    parser.add_argument('--report', type=str, help='Output file for detailed report (supports .csv or .csv.gz)')
    parser.add_argument('--archive-delay', type=float, default=1.0, help='Longer delay for archive.org requests to be respectful')
    
    args = parser.parse_args()
    
    # Load the dataframe - handle both regular CSV and gzipped CSV
    if args.input.endswith('.gz'):
        df = pd.read_csv(args.input, compression='gzip')
        logging.info(f"Loaded compressed CSV file: {args.input}")
    else:
        df = pd.read_csv(args.input)
        logging.info(f"Loaded CSV file: {args.input}")
    
    # If force redownload is set, create a list of files to delete
    if args.force_redownload:
        output_path = Path(args.output_dir)
        if output_path.exists():
            logging.warning(f"Force redownload enabled - clearing existing files in {args.output_dir}")
            for html_file in df['html_file']:
                # Check both regular and gzipped versions
                file_paths = [
                    output_path / html_file,
                    output_path / (html_file + '.gz')
                ]
                for file_path in file_paths:
                    if file_path.exists():
                        try:
                            file_path.unlink()
                            logging.debug(f"Deleted: {file_path}")
                        except Exception as e:
                            logging.error(f"Failed to delete {file_path}: {e}")
    
    # Run the scraper
    main(df, output_dir=args.output_dir, workers=args.workers, batch_size=args.batch_size, 
         delay=args.delay, try_archive=not args.no_archive, archive_timestamp=args.archive_timestamp,
         output_report=args.report)
