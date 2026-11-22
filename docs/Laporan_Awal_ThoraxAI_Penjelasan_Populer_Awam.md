# MENGENAL THORAXAI: ASISTEN DOKTER DIGITAL PENDETEKSI 14 PENYAKIT DADA LEWAT FOTO RONTGEN
## Penjelasan Sederhana & Ramah Mengenai Teknologi Kecerdasan Buatan (AI) untuk Kesehatan Paru-Paru dan Jantung

**Disusun untuk:** Laporan Publik & Pembaca Umum / Non-IT  
**Nama Sistem:** ThoraxAI (Multi-Label Diagnostic Engine)  
**Tahun:** 2026  

---

### KATA PENGANTAR: DILEMA DI RUANG RONTGEN RUMAH SAKIT
Bayangkan sebuah rumah sakit umum yang sangat sibuk. Setiap hari, ratusan bahkan ribuan pasien mengantre untuk difoto rontgen dadanya (*Chest X-Ray*). Ada pasien yang mengeluh batuk berbulan-bulan, sesak napas, nyeri dada, hingga korban kecelakaan. Di sisi lain, dokter spesialis yang bertugas membaca foto hitam-putih tersebut (disebut dokter radiologi) jumlahnya sangat terbatas.

Membaca satu foto rontgen membutuhkan konsentrasi tingkat tinggi. Dokter harus memeriksa bayangan tulang, batas jantung, hingga serat-serat halus di dalam paru-paru. Ketika harus memeriksa ratusan foto dari pagi sampai malam, kelelahan mata dan pikiran tak terelakkan, sehingga ada risiko bercak penyakit kecil terlewatkan.

Inilah alasan proyek **ThoraxAI** diciptakan: **bukan untuk menggantikan peran dokter**, melainkan menjadi **"asisten pintar tanpa lelah"** yang membantu dokter menyaring, memeriksa, dan menunjukkan di mana letak kejanggalan pada foto rontgen dalam hitungan detik.

---

### 1. APA SEBENARNYA THORAXAI ITU?
ThoraxAI adalah program komputer canggih yang menggunakan teknologi *Deep Learning* (cabang dari Kecerdasan Buatan / AI). Jika komputer biasa hanya bisa menjalankan instruksi kaku seperti kalkulator, AI pada ThoraxAI diajari cara "melihat" dan "belajar" layaknya mata dan otak seorang dokter muda yang sedang menempuh pendidikan spesialis.

Sistem ini dilatih dengan melihat **lebih dari 112 ribu foto rontgen dada sungguhan** dari puluhan ribu pasien rumah sakit. Dari ratusan ribu pengalaman visual tersebut, ThoraxAI belajar membedakan mana paru-paru yang bersih dan sehat, serta mana paru-paru yang mengalami infeksi, penumpukan cairan, atau pembesaran organ.

---

### 2. MENGAPA DISEBUT "MULTI-LABEL"? (Satu Orang Bisa Punya Banyak Penyakit)
Dalam aplikasi AI sederhana (misalnya filter foto di ponsel), AI biasanya hanya diminta memilih satu jawaban mutlak: *"Apakah ini foto kucing atau anjing?"*.

Namun, tubuh manusia tidak sesederhana itu. Di dunia nyata, seorang pasien yang sudah lanjut usia atau mengalami komplikasi bisa mengidap beberapa masalah sekaligus. Sebagai contoh:
* Jantungnya mengalami pembengkakan (*Cardiomegaly*),
* Sekaligus paru-parunya terendam cairan (*Effusion*),
* Dan terdapat infeksi aktif (*Infiltration*).

Sistem **Multi-Label** artinya ThoraxAI memeriksa **14 jenis penyakit secara bersamaan dan independen**. AI ini tidak menebak salah satu saja, melainkan memberikan persentase kemungkinan untuk masing-masing dari 14 penyakit tersebut sekaligus, persis seperti dokter yang menuliskan daftar diagnosa lengkap di lembar rekam medis pasien.

---

### 3. SIAPA SAJA "PASIEN" DI BALIK DATA INI? (Informasi Lengkap Dataset NIH)
Agar sebuah kecerdasan buatan bisa pintar, ia membutuhkan buku pelajaran yang sangat tebal dan kredibel. ThoraxAI belajar dari database medis paling terpercaya di dunia, yaitu **NIH ChestX-ray14**, yang dirilis oleh lembaga kesehatan pemerintah Amerika Serikat (*National Institutes of Health*).

#### Fakta Menarik Seputar Data:
1. **112.120 Lembar Foto Rontgen:** Ini setara dengan tumpukan film rontgen setinggi gedung bertingkat jika dicetak fisik.
2. **30.805 Orang Pasien Nyata:** Banyak pasien difoto berkali-kali selama masa perawatan untuk memantau apakah penyakitnya membaik atau memburuk.
3. **Etika & Kerahasiaan Terjaga:** Semua nama, alamat, dan identitas pribadi pasien telah dihapus total (*anonim*) demi menjaga privasi medis.

#### Kamus 14 Penyakit yang Dipelajari oleh ThoraxAI:
Berikut adalah 14 kondisi penyakit dada yang bisa dikenali oleh ThoraxAI, dijelaskan dengan bahasa yang mudah dipahami sehari-hari:

| No | Nama Medis | Bahasa Sehari-hari | Apa yang Terjadi pada Tubuh? | Jumlah Kasus di Data |
|:---:|:---|:---|:---|:---:|
| 1 | **No Finding** | **Sehat / Normal** | Foto rontgen bersih tanpa ada tanda-tanda penyakit yang mencurigakan. | 60.361 foto |
| 2 | **Infiltration** | **Bercak Infeksi Paru** | Ada zat asing (seperti nanah atau lendir) yang mengisi kantung udara paru. | 19.894 foto |
| 3 | **Effusion** | **Paru-Paru Basah** | Penumpukan cairan berlebih di selaput pembungkus paru-paru, membuat pasien sesak. | 13.317 foto |
| 4 | **Atelectasis** | **Paru Mengempis** | Sebagian kantung udara paru mengempis atau tidak mengembang sempurna. | 11.559 foto |
| 5 | **Nodule** | **Benjolan / Flek Kecil** | Titik bulat padat berukuran kecil (kurang dari 3 cm), bisa berupa bekas infeksi atau tumor jinak. | 6.331 foto |
| 6 | **Mass** | **Benjolan / Tumor Besar** | Gumpalan jaringan padat berukuran besar (lebih dari 3 cm) yang memerlukan pemeriksaan biopsi lanjutan. | 5.782 foto |
| 7 | **Pneumothorax** | **Bocor Udara di Rongga Dada** | Paru-paru bocor sehingga udara keluar dan menekan paru hingga sulit bernapas. | 5.302 foto |
| 8 | **Consolidation** | **Paru Memadat** | Jaringan paru yang tadinya berisi udara berubah padat karena radang/radang paru parah. | 4.667 foto |
| 9 | **Pleural Thickening** | **Penebalan Selaput Paru** | Lapisan pembungkus paru menebal membentuk jaringan parut akibat bekas luka masa lalu. | 3.385 foto |
| 10 | **Cardiomegaly** | **Pembengkakan Jantung** | Ukuran bayangan jantung membesar melebihi batas normal separuh rongga dada. | 2.776 foto |
| 11 | **Emphysema** | **Kerusakan Dinding Paru** | Kantung udara paru rusak dan membesar permanen, sering dialami perokok berat. | 2.516 foto |
| 12 | **Edema** | **Cairan Menumpuk Akibat Jantung** | Paru terendam cairan karena pompa jantung melemah (gagal jantung). | 2.303 foto |
| 13 | **Fibrosis** | **Parut Paru-Paru Kaku** | Jaringan paru mengeras dan kaku seperti bekas luka bakar, mengurangi elastisitas napas. | 1.686 foto |
| 14 | **Pneumonia** | **Infeksi Paru-Paru Akut** | Radang paru akibat bakteri/virus yang membuat demam tinggi dan napas berat. | 1.431 foto |
| 15 | **Hernia** | **Organ Perut Naik ke Dada** | Lambung atau usus mencuat naik menembus sekat diafragma ke rongga dada. | 227 foto |

#### Rahasia Penting: Kenapa AI Tidak Boleh "Menyontek Pasien"?
Dalam proyek ini, data dibagi dua: **80% untuk latihan ujian (89.826 foto)** dan **20% untuk ujian kelulusan (22.294 foto)**.

Yang paling penting, pembagian ini dikunci berdasarkan **Orang/Pasien**, bukan sekadar gambar acak. Mengapa? Karena bentuk tulang iga dan tulang selangka setiap manusia itu unik seperti sidik jari! Jika foto Pak Budi di hari Senin dipakai untuk latihan dan foto Pak Budi di hari Kamis dipakai untuk ujian, AI bisa mengenali penyakit Pak Budi hanya karena "menghafal bentuk tulangnya", bukan karena benar-benar paham ilmunya. Dengan memisahkan pasien secara total, kita memastikan AI benar-benar pintar saat menghadapi pasien baru yang belum pernah ia lihat sama sekali.

---

### 4. BAGAIMANA CARA "MATA" AI BEKERJA? (Analogi Detektif Bertingkat)
Bagaimana komputer bisa mengenali penyakit hanya dari foto hitam putih? Komputer menggunakan sistem bernama **Convolutional Neural Network (CNN)**. Cara kerjanya persis seperti cara mata manusia memproses informasi secara bertingkat:

1. **Langkah 1: Melihat Garis dan Tepi (Lapisan Awal)**
   * Seperti anak kecil yang belajar menggambar, di tahap pertama AI hanya mengenali garis lurus dan lengkungan: garis lengkung tulang iga, garis mendatar diafragma, dan garis tepi bahu.
2. **Langkah 2: Mengenali Tekstur (Lapisan Menengah)**
   * AI mulai menggabungkan garis-garis tadi menjadi tekstur: apakah bagian dalam paru terlihat hitam bersih (artinya udara sehat), atau ada corakan putih keruh seperti asap/awan (tanda infeksi/cairan).
3. **Langkah 3: Memahami Bentuk Organ (Lapisan Dalam)**
   * AI mengenali organ secara utuh: bentuk siluet jantung di tengah, kubah paru kiri, dan kubah paru kanan.
4. **Langkah 4: Mengambil Kesimpulan**
   * AI mengukur apakah ukuran jantung terlalu lebar? Apakah sudut di bawah paru-paru lancip atau tumpul terendam air? Dari sinilah nilai kemungkinan penyakit dikeluarkan.

---

### 5. FITUR PENINGKAT KONTRAS CLAHE PADA ANTARMUKA DEMO
Di dunia nyata, foto rontgen tidak selalu memiliki pencahayaan yang sempurna. Ada foto yang terlalu gelap, ada yang terlalu silau putih, atau badan pasien terlalu tebal sehingga rontgen terlihat buram.

Untuk membantu tampilan visual di aplikasi demo web Gradio, ThoraxAI menyediakan tahap peningkatan kontras bernama **CLAHE**.
* **Cara kerja biasa:** Jika kita mencerahkan seluruh foto sekaligus, area yang sudah putih (seperti tulang) akan menjadi silau luar biasa sampai memutih total.
* **Cara kerja CLAHE:** Foto rontgen dibagi menjadi kotak-kotak kecil seperti papan catur (misal 64 kotak). Setiap kotak disesuaikan kontrasnya secara mandiri dan halus sehingga corakan halus di balik tulang rusuk tampak lebih jelas untuk inspeksi visual. *(Catatan teknis: pada iterasi saat ini, model inti dilatih pada citra standar tanpa CLAHE; integrasi CLAHE langsung ke tahap pelatihan menjadi agenda penyempurnaan berikutnya.)*

---

### 6. UJI RESOLUSI KAMERA: APAKAH FOTO PALING BESAR SELALU PALING BAGUS?
Banyak orang mengira: *"Semakin tinggi resolusi foto yang dimasukkan ke model yang sudah jadi, pasti AI akan langsung semakin pintar!"*. Benarkah demikian?

Untuk mengujinya, kami melakukan eksperimen *zero-shot* (menguji model yang dilatih di 224×224 pada 4 tingkat resolusi input tanpa melatih ulang) pada subset 500 foto untuk patologi **`Nodule` (benjolan flek kecil di paru)**:
1. **Resolusi 112×112 (Sangat Buram / Kotak-Kotak):**
   * Kecepatan: Sangat kilat (20,26 milidetik).
   * Hasil: **Menurun (AUC 0,6315)**. Karena terlalu buram, flek kecil ikut hilang menjadi bintik piksel.
2. **Resolusi 224×224 (Selaras dengan Skala Pelatihan):**
   * Kecepatan: Tetap sangat cepat (21,28 milidetik).
   * Hasil: **PALING OPTIMAL (AUC 0,6686)**. Di tingkat ini, ukuran fitur paling cocok dengan yang dipelajari model selama pelatihan, dengan komputasi tetap ringan.
3. **Resolusi 384×384 dan 512×512 (Resolusi Lebih Tinggi Tanpa Pelatihan Ulang):**
   * Kecepatan: Mulai melambat (24,75–28,86 milidetik per foto).
   * Hasil: **Tidak bertambah pintar (AUC 0,6370 dan 0,6315)**. Karena model belum pernah dilatih khusus pada skala tersebut, mengubah ukuran foto terlalu besar saat inferensi saja justru menggeser proporsi pola yang dikenali filter CNN.

**Kesimpulan Penting:** Pada model yang dilatih di 224×224, resolusi **224×224 piksel** adalah titik paling seimbang antara ketajaman diagnosa dan kecepatan inferensi; memanfaatkan resolusi lebih tinggi memerlukan pelatihan ulang (*re-training*) khusus di resolusi tersebut.

---

### 7. AI YANG JUJUR: BISA MENUNJUKKAN "LETAK PENYAKIT" DENGAN GRAD-CAM
Kelemahan terbesar AI zaman dahulu adalah sifatnya yang seperti "kotak hitam ajaib" (*black box*): ia bisa menebak pasien terkena penyakit jantung, tapi ketika ditanya *"Kenapa kamu menebak begitu?"*, komputer tidak bisa menjawab. Hal ini tentu berbahaya di dunia medis.

ThoraxAI dilengkapi teknologi keterbukaan bernama **Grad-CAM**.
* Setelah AI menebak adanya penyakit, sistem ini akan menempelkan **peta warna panas (seperti kamera pengukur suhu tubuh)** di atas foto rontgen:
  * Warna **Merah Menyala**: Bagian yang paling dicurigai dan diperhatikan oleh AI.
  * Warna **Kuning - Hijau**: Bagian pendukung.
  * Warna **Biru Transparan**: Bagian yang dianggap normal dan diabaikan.
* Misalnya, jika AI mendiagnosa **Cardiomegaly (Pembengkakan Jantung)**, warna merah akan menyala tepat di **organ jantung yang membesar**.
* Jika AI mendiagnosa **Effusion (Cairan)**, warna merah akan menyala di **ujung sudut paling bawah rongga dada**.

Dengan cara ini, dokter tidak perlu percaya buta pada AI. Dokter bisa langsung melirik titik merah tersebut dan memverifikasi sendiri: *"Oh benar, di area yang ditunjuk merah ini memang ada cairan!"*.

---

### 8. SEBERAPA PINTAR THORAXAI SAAT INI? (Hasil Ujian Akhir)
Dalam dunia kecerdasan buatan, kepintaran model diukur dengan nilai bernama **ROC-AUC** (nilai 0,5 berarti tebak-tebakan acak seperti melempar koin koin, sedangkan nilai 1,0 berarti sempurna tanpa salah sama sekali).

Setelah menjalani pelatihan selama 5 babak (*epoch*), ThoraxAI berhasil meraih nilai kelulusan rata-rata:
$$\text{Mean ROC-AUC: } \mathbf{0.8326 \text{ (83,26\%)}}$$

Artinya, dalam lebih dari 83% pengujian pada ribuan pasien baru, ThoraxAI mampu membedakan dengan tepat pasien yang sakit dan pasien yang sehat di 14 jenis penyakit yang berbeda.

---

### 9. MANFAAT MASA DEPAN UNTUK RUMAH SAKIT & KITA SEMUA
Bayangkan masa depan fasilitas kesehatan dengan hadirnya sistem seperti ThoraxAI:
1. **Triase Darurat di UGD:** Begitu foto rontgen pasien kecelakaan selesai diambil, dalam waktu kurang dari 1 detik AI langsung memberi peringatan merah: *"Waspada: Terdeteksi Pneumothorax (Paru Bocor) 90%!"*. Pasien kritis bisa langsung dioperasi tanpa menunggu antrean lembar bacaan radiolog.
2. **Pemerataan Akses di Puskesmas Terpencil:** Di pulau atau daerah yang belum memiliki dokter spesialis radiologi, dokter umum setempat bisa menggunakan aplikasi ThoraxAI sebagai "teman konsultasi kedua" untuk membaca foto rontgen pasien sebelum memutuskan rujukan.
3. **Meringankan Beban Tenaga Medis:** Dokter tidak perlu lagi menghabiskan waktu berjam-jam memelototi foto rontgen yang 100% normal. Waktu dan energi dokter bisa difokuskan penuh untuk menangani pasien-pasien yang benar-benar membutuhkan penanganan darurat.

---

### KESIMPULAN RAMAH
ThoraxAI adalah bukti nyata bagaimana teknologi kecerdasan buatan dapat diarahkan untuk misi kemanusiaan yang mulia. Dengan menggabungkan pembelajaran dari 112 ribu foto rontgen, kacamata penjelas kontras, dan kemampuan untuk "menunjuk letak penyakit secara transparan", proyek ini menjadi langkah awal yang kokoh menuju pelayanan kesehatan yang lebih cepat, akurat, dan merata untuk siapa saja.
