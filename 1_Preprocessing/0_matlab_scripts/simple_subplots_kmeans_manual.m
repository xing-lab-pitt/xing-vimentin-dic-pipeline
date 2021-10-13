%dir_path='/home/dante/programs/dvim/user_data/projects/dry/1_dry_project/data/40x/a549_vim_rfp_5mM_calcein_40x/tiled_3x3_tifs';

%seg_path='/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/seg/a549_vim_rfp_2ng_24hr_091721';

%dir_path='/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/tifs/a549_vim_rfp_2ng_24hr_091721/';

%video_path='/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/videos/a549_vim_rfp_2ng_24hr_091721/a549_vim_rfp_2ng_24hr_XY1_091721.avi';

%crop_file='/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/crops/a549_vim_rfp_2ng_24hr_091721/crop.txt';

%crop_file='crop.txt';
%Img_str='a549_vim_rfp_2ng_24hr_091721_T01_XY1_';

%% for tiles
experiment='a549_vim_rfp_control_091621/';
Img_str=strcat(erase(experiment,'/'),'_T01_XY1');
video_name='a549_vim_rfp_control_XY1_091621.avi';
crop_file=strcat(erase(experiment,'/'),'_crop_file.txt');
%tiles index starts with 0
tile='tile0';



dir_path=strcat('/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/tiles/',experiment);
seg_path=strcat('/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/seg/',experiment);
crop_path=strcat('/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/crops/',experiment',crop_file);
video_path=strcat('/home/dap182/cluster/xing/image_analysis/image_data/40x_large_calcein_time_lapse_training_datset/videos/',experiment,video_name);


dic=imread(strcat(dir_path,Img_str,'_C1_',tile,'.tif'));
calcein=imread(strcat(dir_path,Img_str,'_C2_',tile,'.tif'));
vim=imread(strcat(dir_path,Img_str,'_C3_',tile,'.tif'));

%%
implay(video_path)

Img0=imfuse(imadjust(dic),calcein*0.001);%,'blend'

%%
%-------background correction with gaussian filter
Iblur = imgaussfilt(calcein, 300);
%Iblur=medfilt2(I,[150 150]); 
figure
subplot(1,3,1)
imshow(imadjust(calcein));
title('Original Image');
subplot(1,3,2)
imshow(imadjust(Iblur))
title('Gaussian filtered image')
subplot(1,3,3)
I_corr=calcein-Iblur+mean(mean(Iblur));
imshow(imadjust(I_corr))
title('correct Image')
%%
%-------k-means cluster for thresholding-----------
ab = I_corr;
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,1);

nColors =3;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','cityblock', ...
                                      'Replicates',10);
pixel_labels = reshape(cluster_idx,nrows,ncols);
figure
set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
imshow(pixel_labels,[]);


mask=pixel_labels>2;
mask=bwmorph(mask,'fill');
mask=bwmorph(mask,'open');

figure

set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
imshow(mask,[]), title('mask');

Cell_bianyuan= boundarymask(mask);
Cell_bianyuan= bwmorph(Cell_bianyuan,'thin',Inf);
show_boundary = imoverlay(imadjust(dic),Cell_bianyuan,'green');
%figure
%set(gcf, 'Position', get(0, 'Screensize'));
%imshow(show_boundary, 'InitialMagnification','fit');
%imwrite(uint16(Cell_bianyuan),strcat(dir_path,'/kmeans_boundary.png'));

%%
close all

figure
subplottight(2,3,1)
imshow (imadjust(dic), 'border','tight')
title('DIC')
subplottight(2,3,2)
imshow (imadjust(vim), 'border','tight')
title('Vimentin');
subplottight(2,3,3)
imshow (imadjust(calcein), 'border','tight')
title('Calcein');
subplottight(2,3,4)
imshow(pixel_labels,[],'border','tight');
subplottight(2,3,5)
imshow(mask,'border','tight')
title('mask');
subplottight(2,3,6)
imshow(imadjust(I_corr),'border','tight')
title('correct Image')

% fig=figure;
% sp1=subplot(1,3,1,'Parent',fig);
% sp1.Position = sp1.Position + [-0.05 0 0.1 0.1];
% imshow(imadjust(calcein),'border','tight');

% sp2=subplot(1,3,2,'Parent',fig);
% sp2.Position = sp2.Position + [0 0 0.1 0.1];
% imshow(imadjust(vim),'border','tight')
% title('Vimentin')
% sp3=subplot(1,3,3,'Parent',fig);
% sp3.Position = sp3.Position + [0.05 0 0.10 0.1];
% imshow(pixel_labels,[],'border','tight');


%%
%---------------manual draw lines---------------
size_thres=400;
zoom_range=1.5;


figure
%imshow(show_boundary, 'InitialMagnification','fit');
imshow(show_boundary, 'border','tight','InitialMagnification','fit');
%imshow(show_boundary);
%truesize(show_boundary);
set(gcf, 'Position', get(0, 'Screensize'));
%zoom(zoom_range)


h = imfreehand( gca ); setColor(h,'red');

if sum(sum(createMask(h)))>size_thres%
    position = wait( h );
end
while ~isvalid(h)||sum(sum(createMask(h)))==0
    h = imfreehand( gca ); setColor(h,'red');
    if sum(sum(createMask(h)))>size_thres
        position = wait( h );
    end
end

BW = createMask( h );
%mask=mask&~BW;
mask=(mask|BW)&~boundarymask(BW);%boudary is deleted from mask to segment objects
mask = bwareaopen(mask,size_thres);


while sum(BW(:))==0||sum(BW(:)) > size_thres % less than threshold is considered empty mask
    
    % ask user for another mask
    %figure
    Cell_bianyuan= boundarymask(mask);
    show_boundary = imoverlay(imadjust(dic),Cell_bianyuan,'green');
    imshow(show_boundary, []);
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
    mask=(mask|BW)&~boundarymask(BW);
    mask = bwareaopen(mask,size_thres);
end
%%
mask = bwareaopen(mask,size_thres);

seg_mask=bwlabel(mask);

L=label2rgb(seg_mask);
figure
imshow(L,'InitialMagnification','fit');
imwrite(uint16(seg_mask),strcat(seg_path,'/ori_seg_',Img_str,'_',tile,'.png'));

figure
Cell_bianyuan= boundarymask(seg_mask);
show_boundary = imoverlay(Img0,Cell_bianyuan,'green');
imshow(show_boundary, 'InitialMagnification','fit');

crop_list={};
crop_site=0;
    while(crop_site>=0)
        [pick_data, rect] = imcrop(show_boundary);
        %disp(size(pick_data))
        %disp(class(size(pick_data)))
        if isempty(rect)
            break;
        end
        
        %disp(prod(size(pick_data(:,:,1))))
            
        %disp((length(dic)*0.9)^2)
        
        %disp(prod(size(pick_data(:,1,1))))
        
        %disp(prod(size(pick_data(1,:,1))))
        
        if prod(size(pick_data(:,:,1))) > (length(dic)*0.9)^2
            
            %disp(strcat('pick_data',size(pick_data(:,:,1))))
            
            %disp(strcat('threshold',length(dic*0.9)^2))
            
            delete(strcat(seg_path,'*',Img_str,'_',tile,'*'))
            close all
            return
        end
        
        if (prod(size(pick_data(:,1,1)))>360) && (prod(size(pick_data(1,:,1)))>360)
            show_boundary = imoverlay(Img0,Cell_bianyuan,'green');
            imshow(show_boundary, 'InitialMagnification','fit');
            
            crop_site=crop_site+1;
            crop_seg_image=imcrop(seg_mask,rect);

            %[crop_seg_image, nb_cell] = relabel_image(crop_seg_image);
            imwrite(uint16(crop_seg_image),strcat(seg_path,'/seg_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));

            crop_phase=imcrop(dic,rect);
            imwrite(crop_phase,strcat(seg_path,'/crop_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));

            crop_fluor=imcrop(calcein,rect);
            imwrite(crop_fluor,strcat(seg_path,'/fluor_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));

            cell_boundary= boundarymask(crop_seg_image);
            imwrite(uint16(cell_boundary),strcat(seg_path,'/boundary_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));

            Seg_bin=crop_seg_image>0;
            imwrite(uint16(Seg_bin),strcat(seg_path,'/colony_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));

            cell_interior=Seg_bin&~cell_boundary;
            imwrite(uint16(cell_interior),strcat(seg_path,'/interior_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));

            BW_dist=bwdist(~cell_interior);
            imwrite(uint16(BW_dist),strcat(seg_path,'/bwdist_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));

            BIB_3class=2*double(cell_boundary)+double(cell_interior);
            imwrite(uint16(BIB_3class),strcat(seg_path,'/BIB_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));

            crop_list{end+1}=rect;
            
        else 
            show_boundary = imoverlay(Img0,Cell_bianyuan,'red');
            imshow(show_boundary, 'InitialMagnification','fit');
            continue
        end

            
        
            
%         crop_site=crop_site+1;
%         crop_seg_image=imcrop(seg_mask,rect);
%         
%         %[crop_seg_image, nb_cell] = relabel_image(crop_seg_image);
%         imwrite(uint16(crop_seg_image),strcat(seg_path,'/seg_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));
%         
%         crop_phase=imcrop(dic,rect);
%         imwrite(crop_phase,strcat(seg_path,'/crop_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));
%         
%         crop_fluor=imcrop(calcein,rect);
%         imwrite(crop_fluor,strcat(seg_path,'/fluor_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));
%         
%         cell_boundary= boundarymask(crop_seg_image);
%         imwrite(uint16(cell_boundary),strcat(seg_path,'/boundary_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));
%         
%         Seg_bin=crop_seg_image>0;
%         imwrite(uint16(Seg_bin),strcat(seg_path,'/colony_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));
%         
%         cell_interior=Seg_bin&~cell_boundary;
%         imwrite(uint16(cell_interior),strcat(seg_path,'/interior_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));
%         
%         BW_dist=bwdist(~cell_interior);
%         imwrite(uint16(BW_dist),strcat(seg_path,'/bwdist_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));
%         
%         BIB_3class=2*double(cell_boundary)+double(cell_interior);
%         imwrite(uint16(BIB_3class),strcat(seg_path,'/BIB_',Img_str,'_',tile,'_cr',num2str(crop_site),'.png'));
%         
%         crop_list{end+1}=rect;
    end
     
%theoretically, there should be a script that deletes all crops if I mess
%up 
if size(crop_list)~=0
    if ~isfile(crop_file)
        fileID=fopen(crop_file,'a');
        fprintf(fileID,'file_name\trealtive_crops\n')
    else
        fileID=fopen(crop_file,'a');
    end
    
    fprintf(fileID,'%s\t',strcat(dir_path,Img_str,'_C1_',tile,'.tif'));
    %fprintf(fileID,'[%f,%f,%f,%f]\t[',rectout);
    fprintf(fileID,'\t[');
    for i= 1:length(crop_list)
        fprintf(fileID,'[%f,%f,%f,%f]',crop_list{i});
    end
    fprintf(fileID,']\n');
    fclose(fileID);
end

function h = subplottight(n,m,i)
    [c,r] = ind2sub([m n], i);
    ax = subplot('Position', [(c-1)/m, 1-(r)/n, 1/m, 1/n]);
    if(nargout > 0)
      h = ax;
    end
end
