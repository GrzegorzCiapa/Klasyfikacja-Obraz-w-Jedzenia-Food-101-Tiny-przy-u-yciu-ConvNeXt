[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1pyh1upOYTEzTxvIPEODHlkw0zBBWWx0S)
# Klasyfikacja Obrazów Jedzenia (Food-101 Tiny) przy użyciu ConvNeXt



**Autor:** Grzegorz Ciapa  
**Technologie:** Python, PyTorch, Torchvision, Scikit-learn, Matplotlib, Seaborn

## Cel Projektu
Celem tego projektu jest zbudowanie, wytrenowanie i ocena modelu głębokiego uczenia zdolnego do klasyfikacji obrazów z podzbioru popularnego zestawu danych Food-101 (Food-101 Tiny). Zastosowano architekturę ConvNeXt, która reprezentuje nowoczesne podejście do konwolucyjnych sieci neuronowych (CNN), z wykorzystaniem pre-treningu i transfer learningu w celu osiągnięcia wysokiej skuteczności przy ograniczonym zbiorze danych.

## Metodyka

1.  **Zestaw Danych:** Wykorzystano `food-101-tiny`, zawierający pomniejszony zbiór klas z oryginalnego Food-101. Dane są automatycznie pobierane i rozpakowywane z pliku zip w środowisku Colab.
2.  **Architektura Modelu:** Zastosowano pre-trenowany model `convnext_large` z biblioteki Torchvision. Warstwa klasyfikatora (ostatnia warstwa w pełni połączona) została zastąpiona nową warstwą dostosowaną do liczby klas w zestawie Food-101 Tiny. Wszystkie wagi modelu są poddawane treningowi (fine-tuning).
3.  **Augmentacja i Przetwarzanie Danych:**
    * **Zbiór Treningowy:** Zastosowano zaawansowaną augmentację, w tym `RandomResizedCrop` (z interpolacją BICUBIC), `RandomHorizontalFlip` oraz `ColorJitter`, aby zwiększyć odporność modelu.
    * **Zbiór Walidacyjny:** Obrazy są zmieniane do stałego rozmiaru.
    * W obu przypadkach zastosowano standardową normalizację ImageNet.
4.  **Trening:**
    * **Optymalizator:** AdamW, ze współczynnikiem uczenia $3 \times 10^{-5}$ i regularyzacją `weight_decay`.
    * **Funkcja Strata:** CrossEntropyLoss.
    * **Harmonogramowanie LRate:** `CosineAnnealingLR` przez 40 epok.
    * **Mixed Precision:** Wykorzystano `torch.cuda.amp.autocast` i `GradScaler` w celu przyspieszenia obliczeń i oszczędzania pamięci VRAM na GPU.
    * Zastosowano wczesne zatrzymanie (Early Stopping) z cierpliwością 7 epok w oparciu o dokładność walidacji.

## Wykorzystano metryki i wizualizacje:
* **Loss & Accuracy Curves:** Wykresy przedstawiające spadek funkcji straty na zbiorze treningowym i walidacyjnym oraz wzrost dokładności walidacji w kolejnych epokach.
* **Macierz Konfuzji (Confusion Matrix):** Szczegółowa wizualizacja błędów klasyfikacji pomiędzy poszczególnymi kategoriami jedzenia na zbiorze walidacyjnym.
* **Dokładność na klasę:** Wykres słupkowy ilustrujący, jak dobrze model radzi sobie z każdą indywidualną klasą.
* **Predykcja na nowych danych:** Skrypt umożliwia wgranie własnych obrazów, przetworzenie ich i wyświetlenie predykcji wytrenowanego modelu wraz z samym obrazem.

## Zawartość Repozytorium (Plik Notatnika)
* **Rozpakowanie i wczytanie danych:** Ładowanie pliku `.zip`, ustawianie struktur folderów.
* **Definicja transformacji:** Ustawianie augmentacji (Train) i transformacji podstawowych (Valid/Inference). Note: występuje nadpisanie transformacji w kodzie. Transformacje definiowane przy obiektach `datasets.ImageFolder` korzystają z prostego `Resize(64,64)`, podczas gdy `train_transform` i `valid_transform` zdefiniowane później, operujące na rozmiarze 1280x1280, nie wydają się być faktycznie przekazywane do `DataLoaderów`.
* **Budowa Modelu:** Inicjalizacja `convnext_large`, modyfikacja klasyfikatora.
* **Pętla Ucząca:** Implementacja pętli trenującej z mixed precision, zapisywanie najlepszego modelu, ewaluacja po każdej epoce.
* **Ewaluacja i Wizualizacja:** Odtworzenie najlepszego modelu, generowanie wykresów strat, dokładności, macierzy konfuzji i predykcji.
* **Predykcja Interaktywna:** Komórka z `files.upload()`, która przyjmuje obraz z dysku użytkownika i zwraca klasę.

---
*Projekt stanowi przykład implementacji nowoczesnej sieci konwolucyjnej do problemu klasyfikacji obrazów przy użyciu środowiska PyTorch.*
