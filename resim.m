%TestImage yolundaki resmi okuyup A değişkenine atıyoruz.
A = imread('C:\Users\erdmg\Masaüstü\indir.jpg'); 

%YuzBulucu adında  yüz bulma nesnesi oluşturuyoruz.
YuzBulucu = vision.CascadeObjectDetector();
%Resmimizde Kaskad yüz bulmayı çalıştırıyoruz.
%Bulunan yüzlerin koordinat değerlerini BBOX şeklinde bir matris olarak alıyoruz.
%BBOX, sınırlayıcı kutu.
BBOX = step(YuzBulucu, A);

ciz = insertObjectAnnotation(A,'rectangle',BBOX,'Face');
%Bulunan koordinat değerlerini dikdörtgen obje olarak nesneye atıyoruz.
imshow(ciz);
%Resim üzerinde objeyi çizdiriyoruz.