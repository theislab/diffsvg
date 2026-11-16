"""Marker-based labels for the k-means clusters in cuprizone_xenium.h5ad (see inspect_clusters.py output)."""

CELLTYPE = {
    "12": "ependymal",  # Foxj1, Spag16 high (some Ttr: choroid contact)
    "18": "choroid",  # Clic6, Ttr
    "15": "oligo", "23": "oligo",  # Mog, Mag, Opalin; 17% cuprizone (depleted)
    "5": "oligo_lesion",  # mixed Mog + Cd68/Trem2 + Gfap, 92% cuprizone
    "3": "microglia",  # homeostatic: Cx3cr1, Tmem119, Siglech
    "19": "microglia",  # activated (Cd68, Trem2, Spp1, Lgals3), 99% cuprizone
    "25": "opc",
}
