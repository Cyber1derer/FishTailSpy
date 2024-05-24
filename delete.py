total_milliseconds = 324725
total_seconds = total_milliseconds // 1000
minutes = total_seconds // 60
seconds = total_seconds % 60
milliseconds = total_milliseconds % 1000

print(f"{minutes:02}:{seconds:02}:{milliseconds:03}")



