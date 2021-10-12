%dir_path='/Volumes/Fusion0_dat/dante_weikang_imaging_data/a549_vim_rfp_5_calcein/';
% channel_str = 'FITC';
%channel_str = 'TRITC';
% dir_path = '/Volumes/Fusion0_dat/dante_weikang_imaging_data/rpe_pcna_p21_72hr_time_lapse'
%dir_path = '/Volumes/Fusion0_dat/dante_weikang_imaging_data/test_fiji_output/singleTiffs'
% seg_path='/Volumes/Fusion0_dat/dante_weikang_imaging_data/a549_vim_rfp_5_calcein/segs';
% seg_path = '/Volumes/Fusion0_dat/dante_weikang_imaging_data/rpe_pcna_p21_72hr_time_lapse/TRITC_segs'
% seg_path = strcat('/Volumes/Fusion0_dat/dante_weikang_imaging_data/rpe_pcna_p21_72hr_time_lapse/', channel_str, '_segs')
%seg_path = strcat('/Volumes/Fusion0_dat/dante_weikang_imaging_data/test_fiji_output/singleTiffs/', channel_str, '_segs')
%mkdir(seg_path);
% pos='16';
% Img_str=strcat('hk2_calcein_trainxy',pos);
% Img=imread(strcat(dir_path,'/hk2_calcein_trainxy',pos,'c3.tif'));
% I=imread(strcat(dir_path,'/hk2_calcein_trainxy',pos,'c2.tif'));
% Img_str='rpe_pcna_p21_72hr_time_lapse_T050_XY3_'; % legacy: note the ending underscore
%Img_str = 'test_rpe-corrected0008'
%Img=imread(strcat(dir_path,'/',Img_str, '.tif'));
%I=imread(strcat(dir_path,'/',Img_str, '.tif'));

% For non background ver.
% Img=imread(strcat(dir_path,'/',Img_str, channel_str, '.tif'));
% I=imread(strcat(dir_path,'/',Img_str, channel_str, '.tif'));
% Img=imread(strcat(dir_path,'/',Img_str,'TRITC.tif'));
% I=imread(strcat(dir_path,'/',Img_str,'TRITC.tif'));

%% for tiles
experiment='a549_vim_rfp_control_091621/';
tile='tile1';
dir_path=strcat('/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/tiles/',experiment);
seg_path=strcat('/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/crops/',experiment);
Img_str='a549_vim_rfp_control_091621_T04_XY1_C2';
I=imread(strcat(dir_path,Img_str,'_',tile,'.tif'));

%% for debug
% Img = eye(200);
% I = eye(200);


% scale image and pixel intensities
% Img = imresize(Img, 2) / 4; 
% I = imresize(I, 2) / 4;

%%
%-------background correction with gaussian filter
size(I)
I_mean = mean(I(:)*1.0, 1);
sd = std(double(I(:)));

%% adjust contrast
% I_adjusted = imadjust(I, [0.02, 0.99]);
handle = imshow(I);
handle = imcontrast(handle);

Iblur = imgaussfilt(I, 100);
%% blur and normalize
Iblur=medfilt2(I,[150 150]);  
figure
I_corr=I-Iblur+mean(mean(Iblur));
subplot(1,3,1)
imshow(I);
title('Original Image');
subplot(1,3,2)
imshow(imadjust(Iblur))
title('Gaussian filtered image')
subplot(1,3,3)
imshow(imadjust(I_corr))
title('correct Image')
% figure;
% imshow(I);

%% thresholding
% level = multithresh(I_corr, 5);
% seg_I = imquantize(I_corr, level);
% figure;
% imshow(seg_I, []);
% size(label2rgb(seg_I))

%%
%-------k-means cluster for thresholding-----------
ab = I_corr;
nrows = size(ab, 1);
ncols = size(ab, 2);
ab = reshape(ab,nrows*ncols,1);

nColors = 5 ;
% repeat the clustering 3 times to avoid local minima

% [cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','cityblock', ...
%                                       'Replicates', 1);
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqeuclidean', ...
                                      'Replicates', 1);

pixel_labels = reshape(cluster_idx,nrows,ncols);
f = figure
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
title('Kmeans')
imshow(pixel_labels,[]);

mask=pixel_labels > 1;
mask=bwmorph(mask,'fill');
mask=bwmorph(mask,'open');

figure

set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
imshow(mask,[]), title('mask');

Cell_bianyuan= boundarymask(mask);
Cell_bianyuan= bwmorph(Cell_bianyuan,'thin',Inf);
show_boundary = imoverlay(imadjust(I),Cell_bianyuan,'green');
figure
set(gcf, 'Position', get(0, 'Screensize'));
imshow(show_boundary, 'InitialMagnification','fit');

%imwrite(uint16(Cell_bianyuan),strcat(dir_path,'/kmeans_boundary.png'));

%%
%---------------manual draw lines---------------
size_thres=200;
zoom_range=1;



global is_exit;
global is_rubber;
is_exit = false;
is_rubber = false;

figure;
% buttonGroup = uibuttongroup;
c = uicontrol('String', 'Exit', 'Position',[10 250 100 30]);
c.Callback = @setExitFlag;
rubberButton = uicontrol('Style', 'togglebutton', 'String', 'Rubber', 'Position',[10 350 100 30]);
rubberButton.Callback = @setRubberFlag;

show_boundary = imoverlay(imadjust(I_corr),Cell_bianyuan,'green');
imshow(show_boundary, 'InitialMagnification','fit');
set(gcf, 'Position', get(0, 'Screensize'));
zoom(zoom_range)
h = imfreehand( gca ); setColor(h,'red');

% if sum(sum(createMask(h)))>size_thres%
%     position = wait( h );
% end
% while ~isvalid(h)||sum(sum(createMask(h)))==0
%     h = imfreehand( gca ); setColor(h,'red');
%     if sum(sum(createMask(h)))>size_thres
%         position = wait( h );
%     end
% end

BW = createMask( h );
%mask=mask&~BW;
mask=(mask|BW)&~boundarymask(BW);%boudary is deleted from mask to segment objects
mask = bwareaopen(mask,size_thres);




ax = gca;
while ~is_exit %sum(BW(:)) > exit_thres % less than threshold is considered empty mask
    % ask user for another mask
    %figure
    Cell_bianyuan= boundarymask(mask);
    show_boundary = imoverlay(imadjust(Img),Cell_bianyuan,'green');
    cla reset;
    imshow(show_boundary, 'Parent', ax);
    axis on;
    %set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    zoom(zoom_range)
    h = imfreehand( gca ); setColor(h,'red');
    if sum(sum(createMask(h)))>size_thres
        position = wait( h );
    end
    while ~isvalid(h)
        h = imfreehand( gca ); setColor(h,'red');
        if sum(sum(createMask(h)))>size_thres
            position = wait( h );
        end
    end
    
    BW = createMask( h );
    %mask=mask&~BW;
    if ~is_rubber
        mask=(mask|BW)&~boundarymask(BW);
    else
        mask=(mask&~BW);
    end
    
    disp 'is rubber:'
    is_rubber
    mask = bwareaopen(mask,size_thres);
end
%%
mask = bwareaopen(mask,size_thres);
seg_mask=bwlabel(mask);

L=label2rgb(seg_mask);
figure
imshow(L,'InitialMagnification','fit');
imwrite(uint16(seg_mask),strcat(seg_path,'/ori_seg_',Img_str,'.png'));

figure
Cell_bianyuan= boundarymask(seg_mask);
show_boundary = imoverlay(imadjust(I),Cell_bianyuan,'green');
imshow(show_boundary, 'InitialMagnification','fit');


crop_site=0;
    while(crop_site>=0)
        [pick_data, rect] = imcrop(show_boundary);
        if isempty(rect)
            break;
        end
        crop_site=crop_site+1;
        crop_seg_image=imcrop(seg_mask,rect);
        %[crop_seg_image, nb_cell] = relabel_image(crop_seg_image);
        imwrite(uint16(crop_seg_image),strcat(seg_path,'/seg_',Img_str,'cr',num2str(crop_site),'.png'));
        
        crop_phase=imcrop(Img,rect);
        imwrite(crop_phase,strcat(seg_path,'/crop_',Img_str,'cr',num2str(crop_site),'.tif'));
        
        crop_fluor=imcrop(I,rect);
        imwrite(crop_fluor,strcat(seg_path,'/fluor_',Img_str,'cr',num2str(crop_site),'.tif'));
        
        
        cell_boundary= boundarymask(crop_seg_image);
        imwrite(uint16(cell_boundary),strcat(seg_path,'/boundary_',Img_str,'cr',num2str(crop_site),'.png'));
        
        Seg_bin=crop_seg_image>0;
        imwrite(uint16(Seg_bin),strcat(seg_path,'/colony_',Img_str,'cr',num2str(crop_site),'.png'));
        
        cell_interior=Seg_bin&~cell_boundary;
        imwrite(uint16(cell_interior),strcat(seg_path,'/interior_',Img_str,'cr',num2str(crop_site),'.png'));
        
        BW_dist=bwdist(~cell_interior);
        imwrite(uint16(BW_dist),strcat(seg_path,'/bwdist_',Img_str,'cr',num2str(crop_site),'.png'));
        
        BIB_3class=2*double(cell_boundary)+double(cell_interior);
        imwrite(uint16(BIB_3class),strcat(seg_path,'/BIB_',Img_str,'cr',num2str(crop_site),'.png'));
        
    end
    
    
function setExitFlag(src,event)
    disp 'called'
    global is_exit
    is_exit = true;
end

function setRubberFlag(src,event)
    disp 'set rubber called'
    global is_rubber
    is_rubber = ~is_rubber;
end
