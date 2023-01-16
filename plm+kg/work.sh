if [ ! -d "./log" ]; then
  mkdir ./log
fi
for file in $(ls); do
  if [[ $file =~ \.out$ ]]; then # [[ $file =~ \.txt$ ]] 匹配以.txt结尾的文件
    mv $file log/$file  # 将文件移动到log文件夹下
    file=${file%.*}            # 去掉文件名后缀
    file=${file#*-}             # 去掉文件名前缀
    scancel $file # 取消作业
    echo "取消作业 $file"
  fi
done
echo "------------------------------------------------------------------------------"
echo "开始提交作业"
sbatch job.sh
squeue -u zcl