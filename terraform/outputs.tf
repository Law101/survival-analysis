output "linux_machine_ip" {
  value = azurerm_linux_virtual_machine.linux_VM.public_ip_address
}