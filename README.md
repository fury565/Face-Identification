# Face Identification
 
Face Identification je nastao kao projekt kolegija Obrada slike i računalni vid.
Služi kako bi se osoba pred kamerom usporedila s odabranom slikom sa računala.
Uspoređivanje se provodi na način da se na slici pronađe lice te se taj dio slike
provuće kroz pretreniranu VGGFace resnet50 mrežu kako bi se izvukle značajke te se
za kraj mjeri Euklidska udaljenost između njih kako bi se odredila sličnost slika.

Kako iz slika izvlačimo sve značajke, program je osjetljiv na veće promjene tipa
promjene izraza lica, promjena brade i slične. S obzirom na to, najtočniji rezultati
se dobivaju ako se osoba sa slike nije puno promijenila. Ali jedna od prednosti 
je ta što se za određivanje položaja lica koristi Mediapipe biblioteka za koju
nije potrebno imati dedicated GPU kako bi se postigle zadovoljavajuće performanse.