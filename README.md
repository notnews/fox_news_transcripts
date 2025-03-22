## Fox News Transcripts 2003--2025

We scraped Fox News transcripts from [here](https://www.foxnews.com/transcript). In all, we scraped around ~24k transcripts.

## Data

The final dataset, including the HTML file,s is posted on a [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/Q2KIES)

I scraped the data again in 2025 and the breakdown is as follows:

'|   year |   count |\n
|-------:|--------:|\n
|   2003 |     450 |\n
|   2004 |     365 |\n
|   2005 |     431 |\n
|   2006 |     411 |\n
|   2007 |     304 |\n
|   2008 |     418 |\n
|   2009 |     425 |\n
|   2010 |     314 |\n
|   2011 |     523 |\n
|   2012 |    1019 |\n
|   2013 |     777 |\n
|   2014 |     866 |\n
|   2015 |     890 |\n
|   2016 |     821 |\n
|   2017 |    1259 |\n
|   2018 |    1752 |\n
|   2019 |    5865 |\n
|   2020 |    5995 |\n
|   2021 |    5400 |\n
|   2022 |    6782 |\n
|   2023 |    9585 |\n
|   2024 |    8256 |\n
|   2025 |    1474 |'

## Scripts

1. [Get Transcript URLs](01_get_transcript_urls.ipynb)
2. [Download Transcript HTMLs](02_download_transcripts.ipynb)
3. [Get Text from HTMLs](03_transcript_to_text.ipynb)
4. [Upload to Dataverse](04_upload_to_dataverse.ipynb)
