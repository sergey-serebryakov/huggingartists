import argparse
import asyncio
import platform
import re
import typing as t
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict
from tqdm import tqdm as bar

__all__ = ["create_dataset"]


async def get_song_urls(artist_id: int, api_token: str) -> t.List[str]:
    authorization_header = {"authorization": "Bearer " + api_token}
    urls: t.List[str] = []
    async with aiohttp.ClientSession(headers=authorization_header) as session:
        with bar(total=None) as pbar:
            pbar.set_description("⏳ Searching songs...")
            next_page = 1
            while next_page is not None:
                async with session.get(
                    f"https://api.genius.com/artists/{artist_id}/songs?sort=popularity&per_page=50&page={next_page}",
                    timeout=999,
                ) as resp:
                    response = await resp.json()
                    response = response["response"]
                next_page = response["next_page"]

                for song in response["songs"]:
                    urls.append(song["url"])
                pbar.update(len(response["songs"]))
    return urls


def process_page(html: str) -> str:
    """Meant for CPU-bound workload"""
    html = BeautifulSoup(html.replace("<br/>", "\n"), "html.parser")
    div = html.find("div", class_=re.compile("^lyrics$|Lyrics__Root"))
    if div is None:
        lyrics = ""
    else:
        lyrics = div.get_text()

    lyrics = re.sub(r"(\[.*?\])*", "", lyrics)
    lyrics = re.sub("\n{2}", "\n", lyrics)  # Gaps between verses

    lyrics = str(lyrics.strip("\n"))
    lyrics = lyrics.replace("EmbedShare URLCopyEmbedCopy", "").replace("'", "")
    lyrics = re.sub("[\(\[].*?[\)\]]", "", lyrics)
    lyrics = re.sub(r"\d+$", "", lyrics)
    lyrics = str(lyrics).lstrip().rstrip()
    lyrics = str(lyrics).replace("\n\n", "\n")
    lyrics = str(lyrics).replace("\n\n", "\n")
    lyrics = re.sub(" +", " ", lyrics)
    lyrics = str(lyrics).replace('"', "")
    # lyrics = str(lyrics).replace("'", "")
    lyrics = str(lyrics).replace("*", "")
    return lyrics


async def fetch_page(url: str, session: aiohttp.ClientSession):
    """Meant for IO-bound workload"""
    async with session.get(url, timeout=999) as res:
        return await res.text()


async def process(
    url: str, session: aiohttp.ClientSession, pool: ProcessPoolExecutor, pbar: bar
):
    html = await fetch_page(url, session)
    pbar.update(1)
    return await asyncio.wrap_future(pool.submit(process_page, html))


async def parse_lyrics(urls: t.List[str]):
    pool = ProcessPoolExecutor()
    pbar = bar(total=len(urls))
    async with aiohttp.ClientSession() as session:
        pbar.set_description("⏳ Parsing lyrics...")
        coros = (process(url, session, pool, pbar) for url in urls)
        return await asyncio.gather(*coros)


def create_dataset(
    artist_id: int, api_token: str, save_path: Path, skip_if_exists: bool = True
) -> t.Optional[t.Dict[str, t.Tuple[int, int]]]:
    if save_path.exists():
        if not save_path.is_dir():
            raise FileExistsError(
                f"File system path exists ({save_path}) and is not a directory."
            )
        if skip_if_exists:
            return
        save_path.unlink()
    save_path.mkdir(parents=True, exist_ok=True)

    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.get_event_loop()
    try:
        urls = loop.run_until_complete(get_song_urls(artist_id, api_token))
        lyrics = loop.run_until_complete(parse_lyrics(urls))
    finally:
        loop.close()

    # Need to have the `train` key here to make it compatible with existing HF datasets.
    dataset = DatasetDict({"train": Dataset.from_dict({"text": list(lyrics)})})
    dataset.save_to_disk(save_path)

    return {n: d.shape for n, d in dataset.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Make dataset of artist lyrics downloaded from Genius."
    )
    parser.add_argument("--artist_id", type=int, help="Artist ID in Genius API.")
    parser.add_argument("--token", type=str, help="Genius API token")
    parser.add_argument(
        "--save_path", type=str, help="Path where to save the resulting json file"
    )
    args = parser.parse_args()

    create_dataset(args.artist_id, args.token, args.save_path)


if __name__ == "__main__":
    main()
