
                             
clear all 
clc
%%
Sinogram=dlmread('Sinogram_InVivo.txt'); 
Nx=350; %Splatenanzahl des bild_Matrixes
Ny=400; %%Zeilenanzahl
fovx=450;
fovy=400;
dx=fovx/Nx; %Briete  eines Pixels
dy=fovy/Ny;  %Höhe eines Pixels
dD=750/557; % Detektorbreite,  750 ist länge Detektorarray, 557 ist die Anzahl von Detektoren 
nd=557;
Bild_Matrix=zeros(Nx,Ny); %Erstelle  ein Matrix der Größe ny,nx
%%
% wir gehen über alle pixeln im Bild
for n=1:Nx

    for m=1:Ny
        




% Berechnung der physikalischen Koordinaten (VEKTOR r)

        rx=(-fovx/2+((m-1)*dx)); 
        ry=(fovy/2-((n-1)*dy)); 
%%
        % wir gehen über alle winkeln im Sinogram
        for phi =1:180
            Lateralposition=rx*cosd(phi-1)+ry*sind(phi-1); %P in[mm] 
            Lateralindex=round(557/2)+round( Lateralposition/dD);
            % P Index allerdings für matlab. Also
            %Matlab sieht das Sinogram  wie ein Matrix und, wenn wir zum
            %Beispiel 1 bekommen als index hieß es das kistchen, das gerade
            %ober des Isozentrum. Matlab geht aber zu der ersten Zeile und
            %holt werte dann davon aus und das wäre dann falch deswegen
            %haben wir von der Stelle an angefangen zu addieren bzw. zu
            %subtrahieren je nach welcher Index wir bekommen haben. danach
            %kommen wie an der richtigen Stelle im "Sinogram_Matrix"
 %%          
                
            Grauwert(phi)=Sinogram( Lateralindex,phi);
            % Greuwerte werden herausgeholt
 %%        
            a=sum(Grauwert);
            % und dann addiert
        end
       Bild_Matrix(n,m)=a; % addierte Werte eintragen
    end
    
end
figure(1)
subplot(2,1,1)
imagesc(Bild_Matrix)
title('Verschmiertes Bild')
%colormap gray
             
%%
%filter
for i=1:Nx
    for j=1:Ny
    
        kx=(-fovx/2+((i-1)*dx)); 
        ky=(fovy/2-((j-1)*dy)); 
        B_k= sqrt(kx^2+ky^2);
        k(i,j)= B_k;
    end
end
F_Bild_Matrix= fftshift(fft2(Bild_Matrix));
Bild= abs(ifft2(ifftshift(k.*F_Bild_Matrix)));

subplot(2,1,2)
imagesc(Bild);
title('gefiltertes Bild')
                
%%