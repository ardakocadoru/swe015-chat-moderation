# Error analysis (official test set, n = 3,528)

Served model: INT8 BERTurk, threshold 0.54. Values in the tables are P(offensive).

Example tweets with strong profanity, sexual, political or religious content are not shown in the tables. All counts include every test tweet.

* False positives: **159**, false negatives: **245**
* INT8 BERTurk right & tf-idf wrong: **232**; tf-idf right & INT8 BERTurk wrong: **139**
* Decisions changed by quantization (FP32 vs INT8): **47** (1.33 %)

## False positives of INT8 BERTurk (most confident first)

| tweet | gold | lexicon | tfidf | w2v_cnn | bert_fp32 | bert_int8 |
|---|---|---|---|---|---|---|
| Saatlerdir internetten deri pantolon seçmeye çalışıyorum ama hiç bişeyin bana olup olmayacağından emin olamıyorum siz nasıl kahinlersiniz nasıl kumarbazlarsınız | NOT | 0.00 | 0.66 | 0.10 | 0.99 | 0.98 |
| Tamam mı?? T A M AM... | NOT | 0.00 | 0.20 | 0.75 | 0.98 | 0.98 |
| Çok dertlisin aqam noldu sana?   Ağlayıp, o güldüğü fotoğraflarına bakıyordum yine) | NOT | 0.74 | 0.37 | 0.12 | 0.98 | 0.98 |
| @USER Yaramaz çocuk gibiler gözünü ayırma ayarı yok ya la bunların. | NOT | 0.85 | 0.90 | 0.98 | 0.98 | 0.97 |
| İyi oldu iyi helal olsun denizli bu kendini bilmez oyunculara ve yönetime bir tokat lazımdı. Belki akılları başlarına gelir. | NOT | 0.00 | 0.49 | 0.21 | 0.98 | 0.97 |
| Az önce ablam bişey sordu he dedim yeğenim (2) he yok evet var edepsiz dedi folikasit çocuğu seni | NOT | 0.71 | 0.43 | 0.15 | 0.97 | 0.96 |
| Cidden sinirlendim. Severim filan ama bu kopyacılık yaptıkları anlamını değiştirmez. Biz kopya çeksek olay olur ama... yg oyunlarına kanın kanın... güzel pazarl | NOT | 0.00 | 0.41 | 0.08 | 0.97 | 0.96 |
| D&R'a gittik çok istediğim bir kitap vardı 30 tl, okuyup bitirdim çıktım. Bu beleşçilikle iyi yol kat edilir ya :d | NOT | 0.00 | 0.20 | 0.01 | 0.96 | 0.96 |

## False negatives of INT8 BERTurk (least confident first)

| tweet | gold | lexicon | tfidf | w2v_cnn | bert_fp32 | bert_int8 |
|---|---|---|---|---|---|---|
| Tanışmak isteyen ve sohbet etmek isteyen var mı ?   İyi geceler | OFF | 0.00 | 0.15 | 0.02 | 0.00 | 0.00 |
| O günler geçti canımıniçi, bir daha karşılaşmamak dileğiyle | OFF | 0.00 | 0.10 | 0.01 | 0.01 | 0.00 |
| TELEFON MÜZİĞİ DEĞİLMİŞ KENDİSİ DİNLİYORMUŞ HAYIR SABAHIN 6 SI OTOBÜSÜN YARISI UYUYOR DELİREYİM Mİ İLLA | OFF | 0.00 | 0.16 | 0.17 | 0.00 | 0.01 |
| Senin için yansın ama seni yaksın | OFF | 0.00 | 0.83 | 0.10 | 0.01 | 0.01 |
| Galatasarayın mhk görüşmeleri olumlu geçmiş.Masa da bağlamişlar işi | OFF | 0.00 | 0.51 | 0.10 | 0.01 | 0.01 |
| Sipammmmm yaa    Geldi hem laptopu aldı    Hem de beni esir aldı 😂😂 | OFF | 0.00 | 0.17 | 0.00 | 0.01 | 0.01 |
| anlık seni istemiyorum isimli wp kitabı okuyorum kusmama ramak kaldı | OFF | 0.00 | 0.09 | 0.05 | 0.01 | 0.01 |
| Ay ben yine samimiyetsiz saatlerime geldim en iyisi yatmak | OFF | 0.00 | 0.19 | 0.23 | 0.01 | 0.01 |

## INT8 BERTurk right, tf-idf wrong

| tweet | gold | lexicon | tfidf | w2v_cnn | bert_fp32 | bert_int8 |
|---|---|---|---|---|---|---|
| Sizin yapmacık ilişkilerinizi gördükçe yalnızlığımla gurur duyuyorm 😎Canım kendim🖤     Günaydınnnnnn 😇 | OFF | 0.78 | 0.30 | 0.20 | 0.73 | 0.74 |
| @USER Ağzında yara var, boğazında yara var, dış etlerin hassas, elini ısırıp kanatıyorsun.. Yani o kan kesin ağzında var | NOT | 0.00 | 0.68 | 0.13 | 0.47 | 0.45 |
| Galatasaray maçında penaltı verilmedi her fener maçında şenol güneş bordo mavi giyiniyor biz o yüzden şampiyon olamıyos bırak bu işleride bi git artık yaw | NOT | 0.00 | 0.64 | 0.68 | 0.35 | 0.39 |
| @USER futbol artık alt yapılarda kluplerle abileri amcaları büyükleri bağlantılı olan insanların yuvası olduğu için. Milli takımda paraları olanlar var bu siste | NOT | 0.00 | 0.64 | 0.16 | 0.04 | 0.04 |
| @USER Altındaki var istersen ordan daha net anlarsın karakterimi | NOT | 0.00 | 0.74 | 0.90 | 0.04 | 0.03 |
| kriter maiconsa vah bizim halimize kuyu köpek de maicondan iyi stoper | OFF | 0.00 | 0.24 | 0.57 | 0.50 | 0.61 |
| Arkadaslar beşiktaşta olan yokmu aw vizyonsuz gevşek biri olursa seviniriz bye | OFF | 0.00 | 0.51 | 0.16 | 0.98 | 0.98 |
| @USER Ohhhhhhhhhh iyi oldu çakma kabadayıya | OFF | 0.71 | 0.34 | 0.18 | 0.94 | 0.88 |

## tf-idf right, INT8 BERTurk wrong

| tweet | gold | lexicon | tfidf | w2v_cnn | bert_fp32 | bert_int8 |
|---|---|---|---|---|---|---|
| Senin için yansın ama seni yaksın | OFF | 0.00 | 0.83 | 0.10 | 0.01 | 0.01 |
| Erzurum’a evinde 2-0’dan 2-2 yaptırıyon.   Değil Ersun Yanal, Guardiola gelse ifla olmaz bunlar 😂 | NOT | 0.00 | 0.16 | 0.45 | 0.86 | 0.81 |
| Blackpink'in bu kadar güzel fotoğrafı olmasaydı keşke ikide bir icon değiştirip duruyorum aw çok zor | NOT | 0.00 | 0.08 | 0.00 | 0.52 | 0.57 |
| @USER @USER Aşure yenmez mi yaaa ☺️Ne tuhaf lakırdılar bunlar😂😂 | NOT | 0.00 | 0.25 | 0.51 | 0.65 | 0.55 |
| @USER Avrupa maclarida basladimi bu trabzon maf olur .d.d simdiden yoruldu hamsiler | NOT | 0.00 | 0.30 | 0.31 | 0.64 | 0.64 |
| Benim beynim balik değil tamm mı?   Anlayabiliyorum unutabiliyorum .      Tarihin boktan dersleri hariç sjvwiwbxk | NOT | 0.79 | 0.47 | 0.31 | 0.94 | 0.92 |
| @USER bir tane bile matematik sorusu cozmedim okul baslayinca ne bok yicem bilmiyorum | NOT | 0.83 | 0.36 | 0.98 | 0.94 | 0.94 |
| Çok dertlisin aqam noldu sana?   Ağlayıp, o güldüğü fotoğraflarına bakıyordum yine) | NOT | 0.74 | 0.37 | 0.12 | 0.98 | 0.98 |

## Decisions changed by quantization

| tweet | gold | lexicon | tfidf | w2v_cnn | bert_fp32 | bert_int8 |
|---|---|---|---|---|---|---|
| Cok basit bir numaraydı bu, beni mi hafife aldın yoksa zekan mı elvermedi | OFF | 0.00 | 0.26 | 0.07 | 0.61 | 0.50 |
| O zaman da salak gibi kararsızdım | OFF | 0.00 | 0.88 | 0.95 | 0.37 | 0.55 |
| Blackpink'in bu kadar güzel fotoğrafı olmasaydı keşke ikide bir icon değiştirip duruyorum aw çok zor | NOT | 0.00 | 0.08 | 0.00 | 0.52 | 0.57 |
| kriter maiconsa vah bizim halimize kuyu köpek de maicondan iyi stoper | OFF | 0.00 | 0.24 | 0.57 | 0.50 | 0.61 |
| "@USER adlı kullanıcıyı takip etmen ve @USER adlı kullanıcının Tweetlerini görüntülemen engellendi. Daha fazla bilgi al"      Korkak ergen cevap veremeyince eng | OFF | 0.00 | 0.08 | 0.01 | 0.53 | 0.57 |
| Sow golünü Okan Buruk algısı yapan Aykut Kocaman'a hediye etsin 😂 | NOT | 0.85 | 0.22 | 0.05 | 0.54 | 0.47 |
| Yine yine ve yine hiçbir şey yapmadan sadece 3926263 saat yatmak istediğim zamanlardayız bana bi akıl fikir amen | NOT | 0.00 | 0.18 | 0.03 | 0.27 | 0.54 |
| rae aka iflah olmaz serseri, kırdın beni gidiyorum | NOT | 0.00 | 0.37 | 0.02 | 0.55 | 0.46 |

## Error rate of INT8 BERTurk by phenomenon

| subset | n | error rate |
|---|---|---|
| all tweets | 3528 | 11.5 % |
| contains @USER mention | 1684 | 13.1 % |
| contains '*' censoring | 5 | 20.0 % |
| repeated letters (3+) | 156 | 13.5 % |
| very short (<= 5 words) | 106 | 6.6 % |
