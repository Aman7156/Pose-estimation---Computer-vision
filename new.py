from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory="SoccerNetData")

downloader.password = "s0cc3rn3t"   
downloader.downloadGames(
    files=["1_720p.mkv", "2_720p.mkv"],
    split=["train"],
    
)