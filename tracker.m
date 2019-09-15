function [results] = tracker(p, im, ct_area, bg_area, fg_area, area_resize_factor)
%   AMCF: Augmented Memory for Correlation Filters in Fast UAV Tracking

    %% INITIALIZATION
    temp = load('w2crs');
    w2c = temp.w2crs;
    num_frames = numel(p.img_files);
    res_positions = zeros(num_frames, 4);
	pos = p.init_pos;
    target_sz = p.target_sz;
	% Patch of the target + padding
    patch_padded = getSubwindow(im, pos, p.norm_bg_area, bg_area);
    % Initialize hist model
    new_pwp_model = true;
    [bg_hist, fg_hist] = updateHistModel(new_pwp_model, patch_padded, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence);
    bg_hist=single(bg_hist);
    fg_hist=single(fg_hist);
    new_pwp_model = false;
    K=p.max_num_view;
    num_view=0;
    weight_lr=p.weight_lr;
    % Hann (cosine) window
    hann_window = single(myHann(p.cf_response_size(1)) * myHann(p.cf_response_size(2))');
    % Context  suppression window
    context_width=ct_area(2);
    context_height=ct_area(1);
    Q=-context_width/2:context_width/2;
    P=-context_height/2:context_height/2;
    [P, Q]= ndgrid(P,Q);
    w=single(2*(P/context_height).^2+2*(Q/context_width).^2);
    w=mexResize(w, [p.cf_response_size(1),p.cf_response_size(2)] ,'auto');
    
    % Gaussian-shaped desired responses initialization, centred in (1,1)
    % Bandwidth proportional to target size
    output_sigma = sqrt(prod(p.norm_target_sz)) * p.output_sigma_factor / p.hog_cell_size;
    % Reponses for selceted views
    Y=single(zeros(size(hann_window,1),size(hann_window,2),K));
    for i=1:K
        Y(:,:,i)=single(gaussianResponse(p.cf_response_size,power(p.gaussian_var_view,K-i+1)*output_sigma,power(p.gaussian_peak_view,K-i+1)));
    end
    Yf = fft2(Y); 
    % Response for the start frame
    y1=single(gaussianResponse(p.cf_response_size,p.gaussian_var_first*output_sigma,p.gaussian_peak_first));
    yf1=fft2(y1);
    % Response for the current frame
    yc=single(gaussianResponse(p.cf_response_size,output_sigma,1));
    yfc=fft2(yc);
  
    
    %% SCALE ADAPTATION INITIALIZATION
    % Code from DSST
    scale_factor = 1;
    base_target_sz = target_sz;
    scale_sigma = sqrt(p.num_scales) * p.scale_sigma_factor;
    ss = (1:p.num_scales) - ceil(p.num_scales/2);
    ys = exp(-0.5 * (ss.^2) / scale_sigma^2);
    ysf = ((fft(ys)));
    if mod(p.num_scales,2) == 0
        scale_window = single(hann(p.num_scales+1));
        scale_window = scale_window(2:end);
    else
        scale_window = single(hann(p.num_scales));
    end
    ss = 1:p.num_scales;
    scale_factors = p.scale_step.^(ceil(p.num_scales/2) - ss);
    if p.scale_model_factor^2 * prod(p.norm_target_sz) > p.scale_model_max_area
        p.scale_model_factor = sqrt(p.scale_model_max_area/prod(p.norm_target_sz));
    end
    scale_model_sz = floor(p.norm_target_sz * p.scale_model_factor);
    % find maximum and minimum scales
    min_scale_factor = p.scale_step ^ ceil(log(max(5 ./ bg_area)) / log(p.scale_step));
    max_scale_factor = p.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ target_sz)) / log(p.scale_step));
    
   
%% MAIN LOOP
tic;
t_imread = 0;
for frame = 1:num_frames
    if frame>1
        tic_imread = tic;
        im = imread([p.img_path p.img_files{frame}]);
        t_imread = t_imread + toc(tic_imread);
        %% TESTING step
        % extract patch of size bg_area and resize to norm_bg_area
        im_patch_cf = getSubwindow(im, pos, p.norm_bg_area, bg_area);
        pwp_search_area = round(p.norm_pwp_search_area / area_resize_factor);
        % extract patch of size pwp_search_area and resize to norm_pwp_search_area
        im_patch_pwp = getSubwindow(im, pos, p.norm_pwp_search_area, pwp_search_area);  
        % compute feature map
        xt = getFeatureMap(im_patch_cf, p.feature_type, p.cf_response_size, p.hog_cell_size, w2c);
        % apply Hann window
        xt_windowed = bsxfun(@times, hann_window, xt);
        % compute FFT
        xtf = fft2(xt_windowed);
        % Correlation between filter and test patch gives the response
        % Solve diagonal system per pixel.
        if p.den_per_channel
            hf = hf_num ./ (hf_den + p.lambda1);
        else
            hf = bsxfun(@rdivide, hf_num, sum(hf_den, 3)+p.lambda1);
        end
        % Calculate the correlation filter response with channel reliability
        if p.use_weight_channel
                response_cf_chann = single(ensure_real(ifft2(conj(hf) .* xtf)));
                response_cf=sum(bsxfun(@times, response_cf_chann,...
                                reshape(model_chann_w, 1, 1, size(response_cf_chann,3))), 3);
        else
                response_cf = real(ifft2(sum(conj(hf) .* xtf,3))); 
        end
        
        % Crop square search region (in feature pixels).
        response_cf = cropFilterResponse(response_cf, ...
            floor_odd(p.norm_delta_area / p.hog_cell_size));
        if p.hog_cell_size > 1
            % Scale up to match center likelihood resolution.
            response_cf = mexResize(response_cf, p.norm_delta_area,'auto');
            if p.use_weight_channel
                response_cf = single(response_cf)*size(model_chann_w,2);
            end
        end
        
        [likelihood_map] = getColourMap(im_patch_pwp, bg_hist, fg_hist, p.n_bins, p.grayscale_sequence);
        % (TODO) in theory it should be at 0.5 (unseen colors shoud have max entropy)
        likelihood_map(isnan(likelihood_map)) = 0;
        % each pixel of response_pwp loosely represents the likelihood that
        % the target (of size norm_target_sz) is centred on it
        response_pwp = getCenterLikelihood(likelihood_map, p.norm_target_sz);
            
        %% ESTIMATION
        response = mergeResponses(response_cf, response_pwp, p.merge_factor, p.merge_method);
        [row, col] = find(response == max(response(:)), 1);
        center = (1+p.norm_delta_area) / 2;
        pos = gather(pos + ([row, col] - center) / area_resize_factor);
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];

        %% SCALE SPACE SEARCH
        im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor * scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
        xsf = fft(im_patch_scale,[],2);
        scale_response = real(ifft(sum(sf_num .* xsf, 1) ./ (sf_den + p.lambda_scale) ));
        recovered_scale = ind2sub(size(scale_response),find(scale_response == max(scale_response(:)), 1));
        %set the scale
        scale_factor = scale_factor * scale_factors(recovered_scale);
        
        if scale_factor < min_scale_factor
            scale_factor = min_scale_factor;
        elseif scale_factor > max_scale_factor
            scale_factor = max_scale_factor;
        end
        % use new scale to update bboxes for target, filter, bg and fg models
        target_sz = round(base_target_sz * scale_factor);
        avg_dim = sum(target_sz)/2;
        bg_area = round(target_sz + avg_dim);
        if(bg_area(2)>size(im,2)),  bg_area(2)=size(im,2)-1;    end
        if(bg_area(1)>size(im,1)),  bg_area(1)=size(im,1)-1;    end
        bg_area = bg_area - mod(bg_area - target_sz, 2);
        fg_area = round(target_sz - avg_dim * p.inner_padding);
        fg_area = fg_area + mod(bg_area - fg_area, 2);
        % Compute the rectangle with (or close to) params.fixed_area and
        % same aspect ratio as the target bboxgetScaleSubwindow_v1
        area_resize_factor = sqrt(p.fixed_area/prod(bg_area));
    end
       
    %% TRAINING
    % extract patch of size bg_area and resize to norm_bg_area
    im_patch_bg = getSubwindow(im, pos, p.norm_bg_area, bg_area);
    % extract patch of size ct_area and resize to norm_ct_area
    im_patch_ct = getSubwindow(im,pos,p.norm_ct_area,ct_area);
    %compute feature map, of cf_response_size
    bt = getFeatureMap(im_patch_ct,p.feature_type,p.cf_response_size,p.hog_cell_size,w2c);
    xt = getFeatureMap(im_patch_bg, p.feature_type, p.cf_response_size, p.hog_cell_size, w2c);
    % apply Hann window
    xt = bsxfun(@times, hann_window, single(xt));
    % apply Context Suppression window    
    bt = bsxfun(@times, w, single(bt));
    % compute FFT
    btf = fft2(bt);
    xtf = fft2(xt);
        
    
    %% MEMORY INITIALIZATION
    if frame ==1 
        rect_position = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        % save feature map of the start video frame 
        xtf1=xtf;
        ktf1=conj(xtf1).*xtf1;
        % get the fixed term of numerator and denominator (unrelated to views)
        hf_num_first = p.lambda2*bsxfun(@times,conj(yf1),xtf1);
        hf_den_first = p.lambda2*ktf1;
        
        % set the patch of the start frame to the patch of the last view
        last_view_patch=im_patch_bg;
        [lastMatrix,lastSize]=hashing(last_view_patch, p.filterSize);
        % initialization the number of views
        num_view=1;
        % initialization the memory space of views
        xtfn = single(zeros([size(xtf) K]));
        ktfn = single(zeros([size(xtf) K]));
        % add the start frame to the head position of memory space
        xtfn(:,:,:,1)=xtf;
        % get the term ralated to views of numerator and denominator
        for i=1:num_view
            hf_num_view=p.lambda2*bsxfun(@times, conj(Yf(:,:,K-num_view+i)), xtfn(:,:,:,i));
        end
        for i=1:num_view
        ktfn(:,:,:,i)=conj(xtfn(:,:,:,i)).*xtfn(:,:,:,i);
        end
        hf_den_view= p.lambda2*sum(ktfn,4);
    end
    %% MEMORY UPDATE
    % for the visualization of updating views and term update   
    update_view=0;
    % calculate the difference score between the last selected view and the current frame
    % for judging whether to update views
    dif_score=PHA(im_patch_bg,lastMatrix,lastSize,p.filterSize);
    if dif_score>1/2
        % for the visualization of updating views and term update   
        update_view=1;
        % update the patch of the last selected view
        last_view_patch=im_patch_bg;
        [lastMatrix,lastSize]=hashing(last_view_patch, p.filterSize);
        % if the memory space is not full
        % current patch enter the first empty position of the memory space
        if num_view<K
            xtfn(:,:,:,num_view+1)=xtf;
            num_view=num_view+1;
        % if the memory space is full
        % update the memory space like a queue with the earlist view get out 
        else
            for i=1:K-1
                xtfn(:,:,:,i)=xtfn(:,:,:,i+1);
            end
            xtfn(:,:,:,K)=xtf;
        end
    end
    % update the term ralated to views of numerator and denominator
    if  update_view
        for i=1:num_view
            hf_num_view=p.lambda2*bsxfun(@times, conj(Yf(:,:,K-num_view+i)), xtfn(:,:,:,i));
        end
        for i=1:num_view
        ktfn(:,:,:,i)=conj(xtfn(:,:,:,i)).*xtfn(:,:,:,i);
        end
        hf_den_view= p.lambda2*sum(ktfn,4);
    end
           
    %% FILTER UPDATE
        new_hf_num=zeros(size(xtfn,1),size(xtfn,2),size(xtfn,3));
        new_hf_den=zeros(size(xtfn,1),size(xtfn,2),size(xtfn,3));
        % add the fixed term related to the start video frame
        new_hf_num=new_hf_num+hf_num_first;
        new_hf_den=new_hf_den+hf_den_first;
        % add the term related to the selected views in memory space
        new_hf_num=new_hf_num+ hf_num_view;
        new_hf_den=new_hf_den+hf_den_view;
        % add the term related to the current patch and processed context patch
        new_hf_num=new_hf_num+conj(yfc).*xtf;
        new_hf_den =new_hf_den+conj(xtf).*xtf+p.lambda3*conj(btf).*btf;
        if frame == 1
            % first frame, train with a single image
		    hf_den = new_hf_den;
		    hf_num = new_hf_num;
		else
		    % subsequent frames, update the model by linear interpolation
        	hf_den =  (1 - p.learning_rate_cf) * hf_den + p.learning_rate_cf * new_hf_den;
	   	 	hf_num = (1 - p.learning_rate_cf) * hf_num + p.learning_rate_cf * new_hf_num;
            % BG/FG MODEL UPDATE
            % patch of the target + padding
            [bg_hist, fg_hist] = updateHistModel(new_pwp_model, im_patch_bg, bg_area, fg_area, target_sz, p.norm_bg_area, p.n_bins, p.grayscale_sequence, bg_hist, fg_hist, p.learning_rate_pwp);
        end
    %% UPDATE CHANNEL WEIGHT 
    if p.use_weight_channel
        % calculate per-channel feature weights
        if frame==1
            model_chann_w=zeros(1,size(xtf,3));
            model_chann_w(:)=1/size(xtf,3);
        else
            response_cf_channel = ensure_real(ifft2(conj(hf) .* xtf));
            chann_w = max(reshape(response_cf_channel, [size(response_cf_channel,1)*size(response_cf_channel,2), size(response_cf_channel,3)]), [], 1) .* ones(1,size(xtf,3));
            chann_w = chann_w / sum(chann_w);
            model_chann_w = (1-weight_lr)*model_chann_w + weight_lr*chann_w;
            model_chann_w = model_chann_w / sum(model_chann_w);
        end
    end   
      
    %% SCALE UPDATE
    im_patch_scale = getScaleSubwindow(im, pos, base_target_sz, scale_factor*scale_factors, scale_window, scale_model_sz, p.hog_scale_cell_size);
    xsf = fft(im_patch_scale,[],2);
    new_sf_num = bsxfun(@times, ysf, conj(xsf));
    new_sf_den = sum(xsf .* conj(xsf), 1);
    if frame == 1
        sf_den = new_sf_den;
        sf_num = new_sf_num;
    else
        sf_den = (1 - p.learning_rate_scale) * sf_den + p.learning_rate_scale * new_sf_den;
        sf_num = (1 - p.learning_rate_scale) * sf_num + p.learning_rate_scale * new_sf_num;
    end
    
    rect_position_padded = [pos([2,1]) - bg_area([2,1])/2, bg_area([2,1])];
    res_positions(frame,:) = rect_position;
        
    %% VISUALIZATION
    if p.visualization == 1
        figure(1)
        imshow(im);
        if update_view
            text(120,20,['Frame: ' num2str(frame) '/' num2str(num_frames)],'horiz','center','color','w','fontsize',20,'LineWidth',2);
            text(430,20,'New View Selected','horiz','center','color','y','fontsize',20,'LineWidth',2);
            rectangle ('Position',rect_position, 'LineWidth',2, 'EdgeColor','g');
            rectangle('Position',rect_position_padded, 'LineWidth',2, 'EdgeColor','r');
        else
            text(120,20,['Frame: ' num2str(frame) '/' num2str(num_frames)],'horiz','center','color','w','fontsize',20,'LineWidth',2);
            rectangle ('Position',rect_position, 'LineWidth',2, 'EdgeColor','g');
            rectangle('Position',rect_position_padded, 'LineWidth',2, 'EdgeColor','r');
        end
        drawnow
    end
end

    elapsed_time = toc;
    results.type = 'rect';
    results.res = res_positions;
    results.fps = num_frames/(elapsed_time - t_imread);
end


% Reimplementation of Hann window (in case signal processing toolbox is missing)
function H = myHann(X)
    H = .5*(1 - cos(2*pi*(0:X-1)'/(X-1)));
end


% We want odd regions so that the central pixel can be exact
function y = floor_odd(x)
    y = 2*floor((x-1) / 2) + 1;
end

function y = ensure_real(x)
    assert(norm(imag(x(:))) <= 1e-5 * norm(real(x(:))));
    y = real(x);
end

function hammingDist=PHA(image,currentMatrix,currentSize,filterSize)

[hashMatrix1, hashSize1] =hashing(image, filterSize);
hashMatrix2=currentMatrix;
hashSize2=currentSize;
if hashSize1~=hashSize2
    hashSize=min(hashSize1,hashSize2);
else
    hashSize=hashSize1;
end
%Converting the 2D matrix to 1D vector for has
hammingDist1 = num2str(hashMatrix1(:)');
hammingDist2 = num2str(hashMatrix2(:)');
hammingDist =gather(double(sum(hammingDist1~=hammingDist2))/(double(hashSize*hashSize))); 
%finding the Hamming Distance
end

function [hashMatrix, hashSize] =hashing(image, filterSize)
    dim = ndims(image);
    %finding the Dimenstion of the image
    if(dim>2)
    image=rgb2gray(image);			%Converting image into grayscale if RGB
    end
%Required Variables image and filter size for Hashing
    imDCT=dct2(image);
    [~,imcolumn]=size(imDCT);					% size of the DCT matrix
    hashSize=uint32((imcolumn)*(filterSize/100));
    imfiltered=imDCT(1:hashSize,1:hashSize,1);		%Resizing the DCT matrix according to the Filtersize
    imMedian=median(imfiltered(:));					%Finding the median of the filtered matrix
    hashMatrix=imfiltered>=imMedian;				%hash Matrix formed
end

