# Create a temporary filesystem on the RAM (super fast!)
mkdir /tmp/ram                                            # Temporary folder
sudo mount -t tmpfs tmpfs /tmp/ram -o size=8192M          # Creates a 8GB filesystem
dd if=/dev/zero of=/tmp/ram/test.iso bs=1M count=5000     # Tests writing speed
rm /tmp/ram/test.iso                                      # Clears the folder      
sudo umount /tmp/ram                                      # Frees the RAM
