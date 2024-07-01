import pstats

# Carica il file di profilazione
p = pstats.Stats('output_file_200_nodel.prof')

# Ordina per tempo cumulativo e stampa le prime 10 funzioni
p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)