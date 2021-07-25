% Demo to have the user freehand draw an irregular shape over
% a gray scale image, have it extract only that part to a new image,
% and to calculate the mean intensity value of the image within that shape.
% Also calculates the perimeter, centroid, and center of mass (weighted centroid).
% Change the current folder to the folder of this m-file.
clc;	% Clear command window.
clear;	% Delete all variables.
close all;	% Close all figure windows except those created by imtool.
imtool close all;	% Close all figure windows created by imtool.
workspace;	% Make sure the workspace panel is showing.
fontSize = 16;
% Read in a standard MATLAB gray scale demo image.
% folder = fullfile(matlabroot, '\toolbox\images\imdemos');
% baseFileName = 'cameraman.tif';
% % Get the full filename, with path prepended.
% fullFileName = fullfile(folder, baseFileName);
% % Check if file exists.
% % if ~exist(fullFileName, 'file')
% % 	% File doesn't exist -- didn't find it there.  Check the search path for it.
% % 	fullFileName = baseFileName; % No path this time.
% % 	if ~exist(fullFileName, 'file')
% % 		% Still didn't find it.  Alert user.
% % 		errorMessage = sprintf('Error: %s does not exist in the search path folders.', fullFileName);
% % 		uiwait(warndlg(errorMessage));
% % 		return;
% % 	end
% % end
% grayImage = imread(fullFileName);

dir_path='./Img';
seg_path='./Seg/label';

%image_num=20;
size_thres=50;
zoom_range=1;
% pos='18';
% Img_str=%strcat('dicxy',pos);
Img_str='Plate A ch00_B1_1_00d01h00m';
Img=imread(strcat(dir_path,'/',Img_str,'.tif'));

%fig_h = figure;
imshow(imadjust(Img,[],[],1),[]);
axis on;
title('Original Grayscale Image', 'FontSize', fontSize);

%set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
set(gcf, 'units','normalized','outerposition',[0.05 0.05 1 1]);
% message = sprintf('Left click and hold to begin drawing.\nSimply lift the mouse button to finish');
% uiwait(msgbox(message));
%hFH = imfreehand();
zoom(zoom_range)
%pan off
sz=size(Img);
or_mask = false( sz ); % accumulate all single object masks to this one
and_mask = zeros( sz );
seg_mask=zeros(sz);
h = imfreehand( gca ); setColor(h,'red');

%         fig_h = figure;
%         set(fig_h,'KeyPressFcn', 'key_pressed');
%         a=get(fig_h, 'CurrentKey');


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
%resume(h);
i=1;
and_mask=and_mask+double(BW);
or_mask = or_mask | BW; % add mask to global mask
seg_mask=seg_mask+double(BW);



%%
while sum(BW(:))==0||sum(BW(:)) > size_thres % less than 10 pixels is considered empty mask
    
    % ask user for another mask
    %figure
    Cell_bianyuan= boundarymask(seg_mask);
    show_boundary = imoverlay(imadjust(Img) ,Cell_bianyuan,'green');
    %imfuse(imadjust(Img,[],[],0.05) ,Cell_bianyuan);%
    imshow(show_boundary,[]);
    axis on;
    %set(gcf, 'Position', get(0,'Screensize')); % Maximize figure.
    set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
    %zoom(zoom_range)
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
    if sum(BW(:)) > size_thres
        i=i+1;
        
        and_mask=double(or_mask)+double(BW);
        and_mask=double(and_mask>1);
        %             figure
        %             imshow(and_mask,'InitialMagnification','fit');
        
        or_mask = or_mask | BW; % add mask to global mask
        %             figure
        %             imshow(or_mask,'InitialMagnification','fit');
        
        % you might want to consider removing the old imfreehand object:
        %delete( h ); % try the effect of this line if it helps you or not.
        seg_mask=(seg_mask+i*double(BW)).*(double(or_mask)-and_mask);
    end
end
% show the resulting mask
%figure; imshow( seg_mask ,'InitialMagnification','fit'); title('multi-object mask');
L=label2rgb(seg_mask);
figure
imshow(L,'InitialMagnification','fit');
%     imwrite(uint16(seg_mask),strcat(seg_path,'\seg_',Img_str.name(1:end-4),'.png'));
%%

figure
Cell_bianyuan= boundarymask(seg_mask);
show_boundary =imoverlay(imadjust(Img),Cell_bianyuan,'green');% imfuse(imadjust(Img,[],[],0.05) ,Cell_bianyuan);%
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
