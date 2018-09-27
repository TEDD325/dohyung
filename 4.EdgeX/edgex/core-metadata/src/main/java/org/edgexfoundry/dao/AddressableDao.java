/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao;

import java.util.List;
import java.util.stream.Collectors;

import org.edgexfoundry.domain.meta.Addressable;
import org.edgexfoundry.domain.meta.Asset;
import org.edgexfoundry.domain.meta.Device;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Utility class around AddressableRepository for accessing Addresables.
 * 
 */
@Component
public class AddressableDao {

  @Autowired
  private AddressableRepository repos;

  @Autowired
  private DeviceRepository deviceRepos;

  @Autowired
  private DeviceServiceRepository deviceServiceRepos;

  public Addressable getById(String id) {
    return repos.findOne(id);
  }

  public Addressable getByName(String name) {
    return repos.findByName(name);
  }

  public Addressable getByIdOrName(Addressable addressable) {
    if (addressable == null)
      return null;
    if (addressable.getId() != null)
      return repos.findOne(addressable.getId());
    return repos.findByName(addressable.getName());
  }

  public boolean isAddressableAssociatedToDevice(Addressable addressable) {
    return !deviceRepos.findByAddressable(addressable).isEmpty();
  }

  public boolean isAddressableAssociatedToDeviceService(Addressable addressable) {
    return !deviceServiceRepos.findByAddressable(addressable).isEmpty();
  }

  public List<Asset> getOwningServices(Addressable addressable) {
    return getAssociatedDevices(addressable).stream().map(Device::getService)
        .collect(Collectors.toList());
  }

  private List<Device> getAssociatedDevices(Addressable addressable) {
    return deviceRepos.findByAddressable(addressable);
  }

}
