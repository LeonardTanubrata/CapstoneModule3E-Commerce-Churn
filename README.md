# CapstoneModule3E-Commerce-Churn

<img src="Customer_Churn_Prediction_Models_in_Machine_Learning copy.png" alt="alt text">

# Context
Shopify adalah suatu perusahaan e-commerce yang menjual berbagai macam produk di website dan aplikasinya. Manajemen eksekutif Shopify ingin mengidentifikasi pelanggan yang hilang atau pelanggan yang tidak berlangganan kembali di platform mereka, Pelanggan yang hilang dapat disebut customer churn. Perusahan e-commerce perlu melakukan evaluasi pelanggan untuk mengetahui customer churn pada perusahaan.

#Problem Statement
Perusahaan ingin mengetahui customer churn atau pelanggan yang tidak menggunakan jasa perusahaan lagi, yang dapat menimbulkan kerugian bagi perusahaan. Oleh karena itu, perusahaan ingin mencari cara agar bisa melakukan prediksi terhadap customer yang memiliki pola akan churn dengan efisien dan cepat, agar sebelum costumer benar-benar churn, perusahaan dapat menjangkau customer tersebut dengan memberikan layanan yang lebih baik, seperti memberikan promo yang menarik. Namun, sebaiknya promo diberikan kepada orang yang tepat. Dengan demikian, perusahaan dapat menghindari kerugian yang akan terjadi akibat memberikan promo yang sia-sia kepada orang yang tidak tepat (customer tidak churn).

# Objective
Berdasarkan permasalahan tersebut, perusahaan ingin memprediksi pelanggan mana yang loyal dan pelanggan mana yang akan churn. Hal ini sejalan untuk meningkatkan potensi keuntungan yang akan diperoleh perusahaan. Karena hal tersebut, pada sesi ini kita akan memprediksi pelanggan yang akan churn dan tidak, dengan menggunakan pendekatan machine learning berdasarkan pola dalam di data yang tersedia.

# Analytics Approach
- Menemukan pola yang membedakan antara customer yang melakukan churn dan yang tidak melakukan churn dengan cara menganalisis data yang ada pada perusahaan.
- Melakukan prediksi pada customer (churn atau tidak churn) dengan membuat model klasifikasi

# Metrix Evaluation
![image.png](https://i0.wp.com/varshasaini.in/wp-content/uploads/2022/10/confusion-matrix.png?w=633&ssl=1)

- **True Positive (TP)**: Customer diprediksi churn dan kenyataannya melakukan churn
- **False Positive (FP)**: Customer diprediksi melakukan churn, namun pada kenyataannya customer tidak melakukan churn
- **False Negative (FN)**: Customer diprediksi tidak melakukan churn, namun pada kenyataannya customer melakukan churn
- **True Negative (TN)**: Customer diprediksi tidak churn dan kenyataannya tidak melakukan churn

**Type 1 error**: False Positive 
- Action: Memberikan promo dengan perkiraan cost sebesar *$100* per customer per tahun. (perkiraan cost akan di jelaskan di conclusion dibawah)
- Konsekuensi: Memberikan promo pada target yang salah, sehingga perusahaan mengeluarkan biaya yang tidak tepat.

 **Type 2 error**: False Negative
- Action: Tidak memberi perhatian pada customer yang actualnya churn tapi kita prediksi tidak akan melakukan churn.
- Konsekuensi: Perusahaan mengalami kerugian $1000 karena kehilangan customer. (perkiraan rugi akan dijelaskan di conclusion dibawah)

Berdasarkan dari kemungkinan kesalahan prediksi di atas, maka model yang dibuat harus dapat mengurangi atau menekan False Negative, namun tetap menjaga agar perusahaan tidak mengeluarkan biaya yang tidak tepat sasaran dalam memberikan promosi (False Positive). Oleh karena itu, metric utama yang digunakan yakni **`F2 score`**.

![image.png](data:image/png;base64,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)

**`F2 score`** adalah salah satu classification evaluation metrics yang mengukur akurasi model dengan mempertimbangkan trade-off antara precision dan recall. F2 score didasarkan pada nilai beta di mana beta lebih besar dari 1 memberikan lebih banyak penekanan pada recall daripada precision.

**`F2 score`** dapat digunakan untuk mengevaluasi performa model klasifikasi di mana kesalahan false negative lebih penting untuk dihindari daripada false positive.


# Data Dictionary
|Feature|Description|
|---|---|
|Tenure|Tenure of a customer in the company|
|WarehouseToHome|Distance between the warehouse to the customer’s home|
|NumberOfDeviceRegistered|Total number of deceives is registered on a particular customer|
|PreferedOrderCat|Preferred order category of a customer in the last month|
|SatisfactionScore|Satisfactory score of a customer on service|
|MaritalStatus|Marital status of a customer|
|NumberOfAddress|Total number of added on a particular customer|
|Complaint|Any complaint has been raised in the last month|
|DaySinceLastOrder|Day since last order by customer|
|CashbackAmount|Average cashback in last month|
|Churn|Churn flag|

# Data Preprocessing

Langkah-langkah dalam proses preprocessing yaitu dengan **impute missing values** (mengisi data yang kosong), **scaling** (melakukan transformasi terhadap data numerik agar antar variabel memiliki skala yang sama), **encoding** (mengubah data kategorikal menjadi data numerikal) yaitu:

**Impute Missing Values**

Menggunakan Simple Imputer dengan nilai Median: 'Tenure', 'WarehouseToHome', 'DaySinceLastOrder'

Pada tahap explore data di atas, diketahui bahwa pada feature 'Tenure', 'WarehouseToHome' dan 'DaySinceLastOrder' terdapat missing values dan datanya tidak berdistribusi normal, sehingga imputasi dilakukan menggunakan nilai Median. Kolom tersebut juga tergolong numerical features yang tidak memiliki hubungan antar features lainnya, terbukti dari nilai corr < 0.5, sehingga imputasi dilakukan menggunakan metode Simple Imputer, dan tidak memanfaatkan feature lainnya untuk memprediksi missing values.

**Scaling**

Menggunakan Robust Scaler: 'Tenure', 'WarehouseToHome', 'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress', 'DaySinceLastOrder', 'CashbackAmount'

Numerical features di atas memiliki data outliers sehingga dapat menggunakan Robust Scaler yang bisa menghandle data outliers. Feature 'Complain' memiliki data 0 dan 1, sehingga tidak perlu dilakukan scaling lagi.

**Encoding**

Menggunakan One Hot Encoder: 'PreferedOrderCat', 'MaritalStatus'

Pada tahap explore data di atas, diketahui bahwa data pada feature 'PreferedOrderCat' dan 'MaritalStatus' merupakan data kategorikal, sehingga perlu dilakukan encoding untuk mengubah data menjadi numerikal. Encoding dilakukan menggunakan metode One Hot Encoder karena categorical features tersebut tidak memiliki tingkatan/urutan/tidak ordinal dan memiliki unique data yang sedikit. Feature 'PreferedOrderCat' memiliki 5 unique data, dan feature 'MaritalStatus' memiliki 3 unique data.


# Summary

Berdasarkan modeling menggunakan algoritma CatBoost:

- Train Set:
   - Before tuning: `0.75`
   - After tuning: `0.76`

- Test Set:
   - Before tuning: `0.7955`
   - After tuning: `0.7926`


Hyperparameter tuning berhasil meningkatkan `F2 Score` pada train set dari model dengan CatBoost sebesar `0.01`. 
lalu Performa model sebelum tuning mendapat hasil prediksi sebesar `0.7955` hampir setara dengan model setelah tuning yaitu `0.7926`.

# Conclusion

Tujuan dari project ini adalah untuk mengetahui prediksi seorang customer apakah akan melakukan churn atau tidak menggunakan jasa perusahaan e-commerce ini lagi. Berdasarkan business problem di atas, diketahui bahwa:

- Model memiliki kemungkinan prediksi yang benar untuk pelanggan yang churn dan tidak churn `(TP+TN)/total) sebesar 86,03%`
- Model memiliki kemungkinan prediksi benar untuk pelanggan yang churn `(TP/(TP+FN)) sebesar 90%`
- Error Rate pada model ini sebesar `13.97%`

**Type 1 error**: False Positive 
- Action: Memberikan promosi dengan perkiraan cost sebesar `*$100* per capita per tahun`. ($1000 * 10% = $100 promo discount)
(referensi discount : https://www.cmswire.com/customer-experience/how-discounts-affect-customer-lifetime-value/)
- Konsekuensi: Memberikan promo pada target yang salah, sehingga perusahaan mengeluarkan biaya yang tidak tepat

 **Type 2 error**: False Negative
- Action: Tidak memberi perhatian pada customer ini yang kita prediksi tidak akan melakukan churn, dan dapat memberikan profit kepada perusahan dengan perkiraan cost `*$1000* per capita per tahun`.
(referensi : https://lp.littledata.io/average/revenue-per-customer) average revenue per customer $111
- Konsekuensi: Perusahaan mengalami kerugian karena kehilangan customer yang memberikan profit bagi perusahaan
<br>
<br>
- Cost FP : $ 100
- Cost FN : $ 1000


**TANPA MENGGUNAKAN MACHINE LEARNING**

Perusahaan e-commerce tidak dapat mengetahui customer yang akan melakukan churn, sehingga perusahaan e-commerce harus memberikan promosi ke semua customer, agar perusahaan tidak kehilangan customer. Ini menyebabkan perusahaan e-commerce harus mengeluarkan biaya yang besar dalam mengimplentasikan strategi promosinya. 

- Pengeluaran perusahaan untuk promosi `(TP+FP+TN+FN): $100 x 981 = $98,100`
- Promosi yang tepat sasaran pada orang yang churn `(TP+FN): $100 X 160 = $16,000`

Sehingga diketahui bahwa perusahaan e-commerce mengeluarkan biaya yang tidak tepat sasaran (biaya promosi untuk customer yang loyal) sebesar: `$98,100 - $16,000 = $82,100`. Biaya tersebut seharusnya dapat ditekan jika menggunakan Machine Learning.

**DENGAN MENGGUNAKAN MACHINE LEARNING**

Biaya yang tidak tepat sasaran di atas, dapat ditekan jika menggunakan Machine Learning, dengan memprediksi customer yang akan melakukan churn. Sehingga biaya promosi dapat difokuskan kepada customer yang akan melakukan churn, berdasarkan dari hasil prediksi dari Machine Learning.

- Pengeluaran perusahaan e-commerce karena salah promosi ke customer loyal `(FP): $100 x 121 = $12,100`
- Perusahaan e-commerce kehilangan customer karena tidak terprediksi akan churn `(FN): $1000 X 16 = $16,000`

Sehingga diketahui bahwa perusahaan e-commerce mengalami kerugian sebesar: `$12,100 + $16,000 = $24,100`

**KERUGIAN MENURUN SETELAH PAKAI MACHINE LEARNING**

- `Kerugian sebelum pakai Machine Learning: $82,100`
- `Kerugian setelah pakai Machine Learning: $24,100`

Dapat disimpulkan bahwa Machine Learning dengan menggunakan algoritma CatBoost setelah tuning dua kali berhasil menurunkan kerugian perusahaan sebesar `70.64%` --> (($82,100 - $24,100) / $82,100)

Machine Learning mampu mengatasi masalah yang ada pada perusahaan e-commerce ini, yaitu untuk melakukan prediksi pada customer yang akan melakukan churn. Algoritma machine learning yang digunakan pada model ini adalah CatBoost yang dilakukan tuning sebanyak dua kali, dengan nilai akurasi model sebesar `79% `menggunakan `F2 score`. Model ini mampu menurunkan kerugian perusahaan sebesar `70.64%`, dengan menekan angka false negative.

Berdasarkan explainable machine learning menggunakan feature importance, faktor-faktor yang mempengaruhi customer melakukan churn adalah Tenure (lama menjadi customer perusahaan), Complaint, Satisfaction Score, dan Cashback Amount.

# Recommendation

**For Business:**

Customer yang melakukan churn dapat menimbulkan kerugian bagi perusahaan. Oleh karena itu, perusahaan e-commerce perlu menyusun strategi agar dapat tetap menjaga kualitas produk dan pelayanan agar dapat menghindari complaint yang masuk. Selain itu, perusahaan e-commerce perlu menyusun strategi agar dapat menciptakan loyalitas pelanggan, baik dengan melakukan inovasi pada produk yang dipasarkan dan memberikan penawaran yang menarik, sehingga tenure customer semakin tinggi.

Serta melakukan survei untuk pengaplikasian diskon seperti referensi ini (https://www.cmswire.com/customer-experience/how-discounts-affect-customer-lifetime-value/) dimana memberikan medium discount lebih efektif membuat customer menjadi loyal daripada memberikan high discount. Dikarenakan berdasarkan hasil dari EDA dan Feature Importance, `Tenure` yang tinggi/lama akan membuat customer tersebut tidak churn.

Perusahaan perlu menggunakan machine learning yang sudah dibuat, agar dapat mengurangi kerugian bagi perusahaan dengan memberikan promosi tepat sasaran kepada customer yang akan melakukan churn.

**For Model:**

- Menambah jumlah feature supaya model bisa lebih belajar dengan baik contoh feature CustomerID untuk mengetahui bahwa data benar-benar duplikat atau bukan.
- Mencoba menambahkan kombinasi parameter lain dalam hyperparameter tuning.
- Perlu feature lain seperti lama pengiriman produk, ketepatan waktu pengiriman, order count untuk melihat berapa banyak barang yang di pesan, HourSpendOnApp untuk melihat berapa lama customer di dalam aplikasi kita, dan lain-lain.
- Mendorong customer untuk mengisi informasi dengan lengkap supaya tidak ada missing value.




