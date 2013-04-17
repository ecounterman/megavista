function GLM_to_SVM(subject,session)

datadir = sprintf('/Volumes/Plata1/Metacontrast/Scans/%s_%s_Session/%s_%s_n', subject, session, subject, session);
cd(datadir)

% load('Stimuli/optseq/condKey.mat');

%MAKE SURE THESE PARAMETERS ARE CORRECT!!
nRuns = 10;
trialsPerRun = 56;
numConds = 7;

tmpOrient = zeros((nRuns*trialsPerRun),1,numConds);
orientations = zeros(nRuns*trialsPerRun/numConds,1,numConds);


for i = 1:nRuns
    %     parfile{i} = sprintf('Stimuli/parfiles/%s_%s_singletrial_run%02d.par',subject,session,i);
    %     fid = fopen(parfile{i});
    %     stimOrder(:,i) = textscan(fid, '%f%f%s');
    %     fclose(fid);
    
    datafile{i} = sprintf('SVM_Analysis/leftV1Targ_meta%d_multiVoxFigData.mat', i);
    load(datafile{i});
    
    if i>1
        allData = [allData figData];
    else
        allData = figData;
    end
end

numVoxels = size(allData(1).tSeries,2);
tmpBetasMatrix = zeros((nRuns*trialsPerRun),numVoxels,numConds);
betasMatrix = zeros((nRuns*trialsPerRun/numConds),numVoxels,numConds);


for k = 1:nRuns
    for t = 1:trialsPerRun
        label = allData(k).trials.label(t);
        cond = str2double(char(strtok(label,'_')));
        trialNum = allData(k).trials.cond(t);
        
        for a = 1:numVoxels
            iBetas(a)=allData(k).glm.betas(1,t,a);
        end
        
        tmpBetasMatrix(trialNum+trialsPerRun*(k-1),:,cond)=iBetas;
        
        z = strfind(allData(k).trials.label(t),'left');
        tmpOrient(trialNum+trialsPerRun*(k-1),1,cond) = cellfun(@isempty,z);
        clear label cond trialNum iBetas z
    end
end

for c = 1:numConds
    betasMatrix(:,:,c) = tmpBetasMatrix(any(tmpBetasMatrix(:,:,c),2),:,c);
    orientations(:,1,c) = tmpOrient(any(tmpBetasMatrix(:,:,c),2),:,c);
end

end

