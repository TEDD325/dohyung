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

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.edgexfoundry.domain.meta.Device;
import org.edgexfoundry.domain.meta.DeviceReport;
import org.edgexfoundry.domain.meta.DeviceService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class DeviceReportDao {

  @Autowired
  private DeviceReportRepository repos;

  @Autowired
  private DeviceRepository deviceRepos;

  public void removeAssociatedReportsForDevice(Device device) {
    List<DeviceReport> deviceReports = repos.findByDevice(device.getName());
    deviceReports.forEach(dr -> repos.delete(dr));
  }

  public DeviceReport getByIdOrName(DeviceReport report) {
    if (report == null)
      return null;
    if (report.getId() != null)
      return repos.findOne(report.getId());
    return repos.findByName(report.getName());
  }

  public DeviceService getOwningService(DeviceReport report) {
    if (report != null && report.getDevice() != null) {
      Device associatedDevice = deviceRepos.findByName(report.getDevice());
      return associatedDevice.getService();
    }
    return null;
  }

  public List<String> getValueDescriptorsForDeviceReportsAssociatedToDevice(String devicename) {
    List<DeviceReport> deviceReports = repos.findByDevice(devicename);
    return deviceReports.stream().flatMap(r -> Arrays.asList(r.getExpected()).stream())
        .collect(Collectors.toList());
  }

}
