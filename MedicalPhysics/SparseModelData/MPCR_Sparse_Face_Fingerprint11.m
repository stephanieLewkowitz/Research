%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
%
% Machine Perception and Cognitive Robotics Laboratory
%
%     Center for Complex Systems and Brain Sciences
%
%              Florida Atlantic University
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
function MPCR_Sparse_Face2
clear all
close all
clc

N=10; %number of patients
M=8; %Number of photos per patient %10*8=80 images in library

lambda=0.01;%LCA threshhold

[Lp1,L1,patient_names,patient_names_key]=load_library_face(N,M);

[Lp2,L2,fingerprint_names,finger_names_key]=load_library_fingerprint(N,M);



datadouble=0;

    if datadouble==1
    

        L=[L1;L2];
    
         Lp=[Lp1;Lp2];
      %  patient_names_key=[patient_names_key patient_names_key]
        
    elseif datadouble==0;
    
        L=L1; % face
        Lp=Lp1;

   
     elseif datadouble==2;
        
         L=L2;
         Lp=Lp2;
         
    end
   


% L=L+0.4*randn(size(L));


% Library size is (15360, 560) which is (# of pixels, image instance)

r=randperm(size(L,2)); %r is a 1D vector (1,560) number of columns in library, random permutation (ex... 23, 145, 430, 215...)
% whos L
% 
% whos Lp




L=L(:,r); % rearranges the order of the columns according to the permutation map r


Lp=Lp(:,r); 



patient_names_key=patient_names_key(r); % rearranges the order of the patient name key according to the same permutation map r. 




%Now the images in the library are randomized (so that not all the images
%of a patient are stacked together) 



% for i=1:size(L,2)
%
%     imagesc(reshape(L(:,i),120,128))
%     colormap(gray)
%     patient_names_key(i)
%     patient_names{patient_names_key(i)}
%     pause
%
% end

L1=L; % set L1 to L, makes backup instance of Library

p1=[]; % establishes blank, adabtable matrix 

t=[]; % establishes blank, adabtable matrix 

for k=(1:size(L,2))% 1:size(L,2) 1:size(L,2)] %LCA loop, tests each picture one at a time (kth picture); (size(L,2))= N*M*2=560
    
    L=L1;  % library copy, from which one picture will be dropped at a time and classified with LCA against the rest of the library
    
    y=L(:,k); % copy image vector k to be classified
    L(:,k)=0; % delete image vector k from library
    
    yp=Lp(:,k);
    
%     y(1:end/2,:)=0;
%     y(end/2+1:end,:)=0;

   % y=y+.80*randn(size(y));
    
    % yp=yp+.80*randn(size(yp));

    
    
    a=LCA(y, L, lambda); % return sparse code 'a' from LCA optimization routine 
    %(function below), given image y, Library L, and threshhold lambda 
    
    
    b=[]; % establishes blank, adabtable matrix 
    
    for j=1:N % loop through all N patients
        
        b=[b sum(abs(a(find(patient_names_key==j))))]; % find the locations 
        %in sparse vector 'a' that correspond with patient 'j', add them up
        % record the sum in b, so that b is a vector of length N. Each
        % entry of b records the total instances a particular patient is
        % represented in the sparse code. For example, patient '#15' is
        % represented in the 15th entry of 'b' with a number [0,1] that
        % reflects the amount of contributions of patient 15 library images
        % to the sparse vector a. This vector b grows in size on each loop,
        % counting each patient one at a time.
        %
        % 
       
        
    end
    
    [b1,b2]=max(b);%b1 is the max percentage of b, b2 is the patient label. 
    
    
    figure(1)
    set(gcf,'color','w');
    
    subplot(3,3,4)
    bar(b)           % subplots bar graph of 'b'. The tallest bar represents 
    %the patient that contributes the most to the sparse vector 'a'
    
    p=[b2 patient_names_key(k)]; % p is composed of the guess from LCA (b2)
    %with the patient key (p). If b2 is the same as p, the algorithm
    %guessed correclty
    
    p1=[p1; p]; % stacks p into a long list of rows with the two columns 
    %of p, which are the LCA guess and the answer. p1 grows on each loop.
    
   
    
    
    subplot(3,3,[1,2,3])
    plot(p1) %draws a line graph in time of the guess/correct answer
     
    %['Class:' patient_names{patient_names_key(k)} '  Test:' patient_names{patient_names_key(b2)}]

            

   if datadouble==0
    
    
    subplot(3,3,5)
    imagesc(reshape(yp,120,128))  % plots face
    colormap(gray)
    

    
    
    elseif datadouble==2
        
        
        subplot(3,3,8)
       imagesc(reshape(yp,120,128))  % plots fingerprint
     colormap(gray)
        
    else
    
    subplot(3,3,5)
    imagesc(reshape(yp(1:end/2,:),120,128))  % plots face
    colormap(gray)
    
     subplot(3,3,8)
    imagesc(reshape(yp(end/2+1:end,:),120,128))  % plots fingerprint
    colormap(gray)

%        subplot(3,3,8)
%        imagesc(reshape(y,120,128))  % plots fingerprint
%       colormap(gray)

    
    end
    
    

    subplot(3,3,7)             
    bar(abs(a))
     %bar graph of absolute value of sparse vector a, changes for for each image
    %visualizes contribution of each library image to sparse vector a, found with LCA
    
  %  subplot(235)
    t=[t b2==patient_names_key(k)];  % histogram of total correct responses, grows with each loop.
    hist(t,2);
    sum(t)/length(t)
    
    subplot(2,3,6)
    bar(sum(t)/length(t))
    
    
%     drawnow()
     
end

end



function [Lp,L,patient_names,patient_names_key]=load_library_face(N,M) %add the face images to library


patient_names={'an2i','at33','boland','bpm','ch4f','cheyer','choon','danieln','glickman','karyadi'};
patient_names_key=[];%ceil((1:(M*N))/M);

L=[]; %Library
Lp=[];

for i =1:size(patient_names,2) % loop through patients
    
    patient_names{i};
    dr1=dir([patient_names{i} '*open.pgm']);  % directory listing of all files that start with the patients name 
    
    f1={dr1.name}; % get only filenames to cell, here we use both the sunglass/normal pictures
    
    D=[]; 
    Dp=[];%clear the Dictionary for each new patient
    
    for j=1:M %loop through images
        
        a1=f1{j};
        
        b1=im2double(imread(a1)); %image to double precision 
        
        b1=b1(1:end)'; % transpose image to column vector
        
        
        b1 = b1 - min(b1(:)); % these two lines normalize the image 
        b1 = b1 / max(b1(:));
        c1= b1;

        D=[D b1];  % add image to dictionary of patient 
        Dp=[Dp c1];  % add image to dictionary of patient 
        
        patient_names_key=[patient_names_key i]; % sets patient name key
        
    end
    
    %image processing routine to whiten the image spectrum
    
%     D = bsxfun(@minus,D,mean(D)); %remove mean
    fX = fft(fft(D,[],2),[],3); %fourier transform of the images
    spectr = sqrt(mean(abs(fX).^2)); %Mean spectrum
    D = ifft(ifft(bsxfun(@times,fX,1./spectr),[],2),[],3); %whitened L
    
    L=[L D]; % The Library is the concatenation of the Dictionary from each patient
    Lp=[Lp Dp]; 
end


% L = bsxfun(@minus,L,mean(L)); %remove mean
 fX = fft(fft(L,[],2),[],3); %fourier transform of the images
 spectr = sqrt(mean(abs(fX).^2)); %Mean spectrum
 L = ifft(ifft(bsxfun(@times,fX,1./spectr),[],2),[],3); %whitened L


end




function [Lp,L,fingerprint_names,finger_names_key]=load_library_fingerprint(N,M)

fingerprint_names={'101','102','103','104','105','106','107','108','109','110'};
finger_names_key=[];%ceil((1:(M*N))/M);

L=[]; %Library

 Lp=[];

for i =1:size(fingerprint_names,2)
    
    fingerprint_names{i};
    
    dr1=dir([fingerprint_names{i} '*.tif']);
    
    f1={dr1.name}; % get only filenames to cell
    
    D=[]; %  Dictionary
    Dp=[];
   
    c1=0;
    
    for j=1:M %  length(f1) % for each image
        
        a1=f1{j};  
        b1=im2double(imread(a1));
        
       
        b1=imresize(b1,[120,128]);
        b1=b1(1:end)';
        
        b1 = b1 - min(b1(:));
        b1 = b1 / max(b1(:));
        c1= b1;
         
       
        D=[D b1];
        finger_names_key=[finger_names_key i];
        
        Dp=[Dp c1];
        
        
    end
    
%   D = bsxfun(@minus,D,mean(D)); %remove mean
    fX = fft(fft(D,[],2),[],3); %fourier transform of the images
    spectr = sqrt(mean(abs(fX).^2)); %Mean spectrum
    D = ifft(ifft(bsxfun(@times,fX,1./spectr),[],2),[],3); %whitened L
    
    
    L=[L D];
    
    Lp=[Lp Dp];
    
    
end

% L = bsxfun(@minus,L,mean(L)); %remove mean
fX = fft(fft(L,[],2),[],3); %fourier transform of the images
spectr = sqrt(mean(abs(fX).^2)); %Mean spectrum
L = ifft(ifft(bsxfun(@times,fX,1./spectr),[],2),[],3); %whitened L


end







function [a, u] = LCA(y, D, lambda) % LCA routine


t=.01; %time step
h=.000001; %step size

d = h/t; %dx/dt
u = zeros(size(D,2),1); %initialize u vector as zeros. We pass the LCA routine L (Library) for D here. size(L,2) is 560


for i=1:100
    
    
    a=u.*(abs(u) > lambda); % a is zero if abs(u) is below lambda threshold
    
    u =   u + d * ( D' * ( y - D*a ) - u - a  ) ; % integrate u with euler interation. As the code updates, the vector 'a' 
    % converges to the sparsest represtation of y with L*a, analogue
    % electronics 
    
    pause(0.22)
    
    figure(4)
    set(gcf,'color','w')
    clf
    
    
    hold on
   
    sub1=subplot(1,3,1) ;            
    bar(abs(a))
    title(sub1,'bar(abs(a))')
       
     
    sub2=subplot(1,3,2);             
    bar(a)
    title(sub2,'bar(a)')
    
    
    hold off

    
    
    sub3= subplot(1,3,3)  ;           
    bar(u)
    title(sub3,'bar(u)')
    
    
end


 
   
end



